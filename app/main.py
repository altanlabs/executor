import json
import logging
import sys
import os
import io
from threading import Lock
import traceback
import tempfile
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Set
import pkg_resources
import site

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Or INFO in production

# Make sure you have a handler if running standalone
if not logger.handlers:
	stream_handler = logging.StreamHandler()
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

# Constants
MAX_EXECUTION_TIME = 300  # seconds
MAX_MEMORY = 256 * 1024 * 1024  # 256MB
DEPENDENCY_PATH = "/mnt/deps"

# This controls how many times a pip installation can be retried if there are transient errors.
MAX_INSTALL_RETRIES = 2

# Global dependency cache and a threading lock for thread safety.
_installed_packages: Set[str] = None
_installed_packages_lock = Lock()

# Initialize FastAPI app
app = FastAPI(
	title="Python Code Executor API",
	description="API for executing Python code dynamically with dependency management",
	version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Ensure dependencies are in the Python path
os.environ["PYTHONPATH"] = DEPENDENCY_PATH

# Create dependency directory if it doesn't exist
os.makedirs(DEPENDENCY_PATH, exist_ok=True)

class ExecuteRequest(BaseModel):
	code: str
	input_vars: Optional[Dict[str, Any]] = {}
	output_vars: Optional[List[str]] = None
	dependencies: Optional[List[str]] = []

class ExecuteResponse(BaseModel):
	result: Optional[Dict[str, Any]] = None
	error: Optional[str] = None
	details: Optional[str] = None

# =========================
# HELPER: GET INSTALLED PACKAGES
# =========================

async def get_installed_packages_async() -> Set[str]:
	"""
	Asynchronously retrieve a set of all installed packages (case-insensitive).
	Includes packages from the default environment and from DEPENDENCY_PATH.
	
	This function uses asyncio.to_thread for CPU-bound or blocking operations.
	That ensures it doesn't block the main event loop.
	
	Returns:
		A set of package names (lowercase).
		
	Raises:
		Exception: If there's an error accessing pkg_resources or the custom path.
	"""
	try:
		logger.debug("Gathering installed packages from default environment via pkg_resources.")
		base_installed = await asyncio.to_thread(
			lambda: {pkg.key.lower() for pkg in pkg_resources.working_set}
		)
	except Exception as exc:
		logger.error("Error accessing pkg_resources.working_set", exc_info=True)
		raise exc

	combined_installed = set(base_installed)

	# Safely check our custom path
	try:
		if os.path.exists(DEPENDENCY_PATH):
			logger.debug("Custom dependency path '%s' exists. Gathering distributions.", DEPENDENCY_PATH)
			# Add path and gather packages asynchronously
			await asyncio.to_thread(site.addsitedir, DEPENDENCY_PATH)
			custom_installed = await asyncio.to_thread(
				lambda: {pkg.key.lower() for pkg in pkg_resources.find_distributions(DEPENDENCY_PATH)}
			)
			combined_installed.update(custom_installed)
		else:
			logger.debug("Custom dependency path '%s' does not exist; skipping custom packages.", DEPENDENCY_PATH)
	except Exception as exc:
		logger.error("Error processing custom dependency path.", exc_info=True)
		raise exc

	logger.debug("Total installed packages found: %d", len(combined_installed))
	return combined_installed


# =========================
# HELPER: RUN PIP INSTALL
# =========================

async def run_pip_install_async(packages: List[str], attempt: int = 1) -> None:
	"""
	Runs pip install asynchronously to install the given list of packages into DEPENDENCY_PATH.
	
	Args:
		packages: A list of packages (exact strings for pip, e.g., ['requests==2.25.1']).
		attempt: Current attempt count for retries.
	
	Raises:
		HTTPException: If the installation fails or times out, or if the user must consult server logs.
	"""
	command = [sys.executable, "-m", "pip", "install", "--target", DEPENDENCY_PATH, *packages]
	logger.info("Attempt [%d] - Executing pip install command: %s", attempt, ' '.join(command))

	# We'll open a subprocess asynchronously
	try:
		# Create the subprocess
		process = await asyncio.create_subprocess_exec(
			*command,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE
		)

		# Wait up to 300 seconds for the pip install to complete
		try:
			stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
		except asyncio.TimeoutError:
			# kill the process to avoid orphaned children
			process.kill()
			await process.communicate()
			logger.error("Timeout during pip install execution on attempt [%d].", attempt, exc_info=True)
			raise HTTPException(
				status_code=408,
				detail={
					'result': None,
					'error': 'Dependency Installation Timeout',
					'details': "Installation timed out. Check server logs for further diagnostics."
				}
			)

		# Convert bytes to string
		out_str = stdout.decode().strip()
		err_str = stderr.decode().strip()

		logger.debug("pip install stdout (attempt [%d]):\n%s", attempt, out_str)
		logger.debug("pip install stderr (attempt [%d]):\n%s", attempt, err_str or "(none)")

		# If the return code is non-zero, we might need to retry or fail.
		if process.returncode != 0:
			logger.error(
				"pip install command failed (attempt [%d]). Return Code: %s | Stdout: %s | Stderr: %s",
				attempt, process.returncode, out_str, err_str,
				exc_info=True
			)
			raise subprocess.CalledProcessError(
				returncode=process.returncode,
				cmd=command,
				output=stdout,
				stderr=stderr
			)

	except HTTPException:
		# Just re-raise any HTTPException for handling in outer scope
		raise
	except subprocess.CalledProcessError as cpe:
		# If we haven't exceeded max retries, we can retry
		if attempt < MAX_INSTALL_RETRIES:
			logger.warning("Retrying pip install (attempt [%d -> %d]) for packages: %s", attempt, attempt + 1, packages)
			await run_pip_install_async(packages, attempt=attempt + 1)
		else:
			logger.error("Max install retries reached. Failing out.")
			raise HTTPException(
				status_code=400,
				detail={
					'result': None,
					'error': 'Dependency Installation Error',
					'details': "Repeated pip errors during installation. Check server logs for specifics."
				}
			)
	except Exception as exc:
		tb = traceback.format_exc()
		logger.critical("Unexpected error in run_pip_install_async (attempt [%d]): %s", attempt, tb)
		raise HTTPException(
			status_code=500,
			detail={
				'result': None,
				'error': 'Unexpected Dependency Installation Error',
				'details': "An unexpected error occurred during installation. Check server logs for traceback."
			}
		)


# =========================
# MAIN INSTALL FUNCTION
# =========================

async def install_dependencies_async(dependencies: List[str]) -> None:
	"""
	Asynchronously installs Python dependencies if not already installed.
	
	Key Features:
	- Validates user input thoroughly.
	- Thread-safe + async-safe initialization of the global cache.
	- Multi-step approach for computing missing dependencies.
	- Uses an async subprocess call with timeouts to avoid locking the event loop.
	- Includes optional retry logic (MAX_INSTALL_RETRIES) for transient network issues.
	- Sanitizes error details for HTTP responses while logging the savage truth.
	
	Raises:
		HTTPException: On any error (with sanitized messages for the client).
	"""
	logger.debug("START: Async dependency installation process. Input: %s", dependencies)

	# Validate input strictly
	if not isinstance(dependencies, list) or not all(isinstance(dep, str) and dep.strip() for dep in dependencies):
		logger.error("Invalid dependencies input. Must be a list of non-empty strings.")
		raise HTTPException(
			status_code=400,
			detail={
				'result': None,
				'error': 'Invalid Dependency Input',
				'details': "Dependencies must be a list of non-empty strings."
			}
		)

	if not dependencies:
		logger.warning("No dependencies provided; skipping installation.")
		return

	# Ensure the dependency path is valid
	if not os.path.isdir(DEPENDENCY_PATH):
		logger.error("Dependency path '%s' is not a valid directory or does not exist.", DEPENDENCY_PATH)
		raise HTTPException(
			status_code=500,
			detail={
				'result': None,
				'error': 'Invalid Dependency Path',
				'details': f"Configured dependency path '{DEPENDENCY_PATH}' is invalid. Check server config."
			}
		)

	global _installed_packages
	# Acquire a thread-level lock for the global cache initialization
	try:
		async with asyncio.Lock():
			if _installed_packages is None:
				logger.debug("Initializing the global installed packages cache asynchronously.")
				_installed_packages = await get_installed_packages_async()
	except Exception as exc:
		logger.error("Cache initialization error", exc_info=True)
		raise HTTPException(
			status_code=500,
			detail={
				'result': None,
				'error': 'Cache Initialization Error',
				'details': "Error initializing package cache. Check server logs."
			}
		)

	# Compute which dependencies are missing (case-insensitive)
	lower_installed = _installed_packages  # already lowercased
	missing = [dep for dep in dependencies if dep.lower() not in lower_installed]

	if not missing:
		logger.info("All dependencies are already installed; nothing to do.")
		return

	logger.info("Missing dependencies: %s", missing)

	# Install the missing dependencies asynchronously (with retries if needed).
	await run_pip_install_async(missing)

	# Once installed, update the global cache again in a thread-safe manner
	async with asyncio.Lock():
		_installed_packages.update(dep.lower() for dep in missing)

	# Ensure that the newly installed packages are in sys.path
	if DEPENDENCY_PATH not in sys.path:
		sys.path.append(DEPENDENCY_PATH)
		logger.debug("Appended dependency path '%s' to sys.path.", DEPENDENCY_PATH)

	logger.info("Successfully installed dependencies: %s", missing)


def prepare_user_code(code: str, input_vars: Optional[Dict] = None, output_vars: Optional[List] = None) -> str:
	# Inject input variables from the dictionary into the user's code
	input_vars_code = "\n".join([f"{key} = {repr(value)}" for key, value in (input_vars or {}).items()])
	output_vars_condition = f"and not (k in {str(list((input_vars or {}).keys()))})" if not output_vars else f" and k in {str(output_vars)}"

	# Wrap user code to capture print statements and extract variables
	wrapped_code = """
import json
import sys
import io
from contextlib import redirect_stdout

class EnhancedJSONEncoder(json.JSONEncoder):
	def default(self, obj):
		try:
			return super().default(obj)
		except TypeError:
			if isinstance(obj, set):
				return list(obj)
			if isinstance(obj, complex):
				return {'real': obj.real, 'imag': obj.imag}
			if isinstance(obj, bytes):
				return obj.hex()
			return str(obj)
"""
	if input_vars_code:
		wrapped_code += "\n# Input variables\n"
		wrapped_code += input_vars_code

	wrapped_code += """
# Buffer for debug messages
debug_output = io.StringIO()

try:
	with redirect_stdout(debug_output):
"""
	# Indent user code
	for line in code.split("\n"):
		wrapped_code += f"        {line}\n"

	wrapped_code += """
	# Filter variables
	variables = {k: v for k, v in locals().items() if not k.startswith('__') and not callable(v) """
	wrapped_code += output_vars_condition
	wrapped_code += " }\n"
	wrapped_code += """
	output = {
		'vars': variables,
		'debug': debug_output.getvalue()
	}
	print(json.dumps(output, cls=EnhancedJSONEncoder))
except Exception as e:
	print(json.dumps({'error': str(e)}), file=sys.stderr)
"""
	return wrapped_code

async def execute_python_code(
	code: str,
	input_vars: Optional[Dict[str, Any]] = None,
	output_vars: Optional[List[str]] = None
) -> Dict[str, Any]:
	try:
		# Add resource limits
		resource_limit_code = f"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, ({MAX_EXECUTION_TIME}, {MAX_EXECUTION_TIME}))
resource.setrlimit(resource.RLIMIT_AS, ({MAX_MEMORY}, {MAX_MEMORY}))
"""
		full_code = resource_limit_code + "\n" + prepare_user_code(code, input_vars, output_vars)

		# Create temporary file
		with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_file:
			tmp_file.write(full_code.encode('utf-8'))
			tmp_file_name = tmp_file.name

		try:
			# Execute as subprocess
			result = subprocess.run(
				['python3', tmp_file_name],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				timeout=MAX_EXECUTION_TIME,
				text=True
			)

			# Clean up
			os.remove(tmp_file_name)

			stripped_stdout = result.stdout.strip() if result.stdout else None
			stripped_stderr = result.stderr.strip() if result.stderr else None

			if result.returncode == 0 and stripped_stdout:
				return {'result': json.loads(stripped_stdout), 'error': None}
			else:
				try:
					error_dict = json.loads(stripped_stderr)
				except Exception:
					error_dict = None
				return {
					'result': None,
					'error': 'Execution Error',
					'details': error_dict['error'] if error_dict else stripped_stderr
				}

		except subprocess.TimeoutExpired:
			os.remove(tmp_file_name)
			return {
				'result': None,
				'error': 'Execution Error',
				'details': 'Execution time limit exceeded'
			}

	except Exception as e:
		return {
			'result': None,
			'error': 'Execution Error',
			'details': traceback.format_exc()
		}

@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
	"""Execute Python code with optional dependencies and input/output variables."""
	if not request.code:
		raise HTTPException(status_code=400, detail="No code provided")

	# Install missing dependencies
	if request.dependencies:
		try:
			await install_dependencies_async(request.dependencies)
		except Exception as e:
			return {
				'result': None,
				'error': 'Dependency Error',
				'details': str(e)
			}

	# Execute user-provided Python code
	result = await execute_python_code(
		request.code,
		request.input_vars,
		request.output_vars
	)

	if result.get('error'):
		raise HTTPException(
			status_code=400,
			detail={
				'error': result['error'],
				'details': result['details']
			}
		)

	return result

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8080) 