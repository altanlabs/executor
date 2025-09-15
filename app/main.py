import json
import logging
import sys
import os
import traceback
import tempfile
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Set
import pkg_resources
import ast
import site
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Imports the Cloud Logging client library
import google.cloud.logging

# Instantiates a client
client = google.cloud.logging.Client()

# Retrieves a Cloud Logging handler based on the environment
# you're running in and integrates the handler with the
# Python logging module. By default this captures all logs
# at INFO level and higher
client.setup_logging(log_level=logging.DEBUG)

BLOCKED_LIBRARIES = {
    "os",
    "subprocess",
    "multiprocessing",
    "threading",
    "concurrent",
    "signal",
    "resource",
    "gc",
    "sys",
    "ctypes",
    "platform",
    "os",
    "subprocess",
    "multiprocessing",
    "threading",
    "concurrent",
    "signal",
    "resource",
    "gc",
    "sys",
    "ctypes",
    "platform",
}


# Constants
MAX_EXECUTION_TIME = 300  # seconds
MAX_MEMORY = 256 * 1024 * 1024  # 256MB
DEPENDENCY_PATH = "/mnt/deps"

# This controls how many times a pip installation can be retried if there are transient errors.
MAX_INSTALL_RETRIES = 2

# Global dependency cache and a threading lock for thread safety.
_installed_packages: Set[str] = None
# _installed_packages_lock = Lock()

# Initialize FastAPI app
app = FastAPI(
    title="Python Code Executor API",
    description="API for executing Python code dynamically with dependency management",
    version="1.0.0",
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
        logging.debug(
            "Gathering installed packages from default environment via pkg_resources."
        )
        base_installed = await asyncio.to_thread(
            lambda: {pkg.key.lower() for pkg in pkg_resources.working_set}
        )
    except Exception as exc:
        logging.error("Error accessing pkg_resources.working_set", exc_info=True)
        raise exc

    combined_installed = set(base_installed)

    # Safely check our custom path
    try:
        if os.path.exists(DEPENDENCY_PATH):
            logging.debug(
                "Custom dependency path '%s' exists. Gathering distributions.",
                DEPENDENCY_PATH,
            )
            # Add path and gather packages asynchronously
            await asyncio.to_thread(site.addsitedir, DEPENDENCY_PATH)
            custom_installed = await asyncio.to_thread(
                lambda: {
                    pkg.key.lower()
                    for pkg in pkg_resources.find_distributions(DEPENDENCY_PATH)
                }
            )
            combined_installed.update(custom_installed)
        else:
            logging.debug(
                "Custom dependency path '%s' does not exist; skipping custom packages.",
                DEPENDENCY_PATH,
            )
    except Exception as exc:
        logging.error("Error processing custom dependency path.", exc_info=True)
        raise exc

    logging.debug("Total installed packages found: %d", len(combined_installed))
    return combined_installed


def validate_library_security(library_name: str) -> bool:
    """
    Validate if a library is safe to install and use.

    Args:
        library_name: Name of the library to validate

    Returns:
        True if library is safe, False otherwise
    """
    # Normalize library name (handle version specifiers)
    normalized_name = re.split(r"[>=<!=]", library_name.lower())[0].strip()
    normalized_name = normalized_name.replace("_", "-").replace(".", "-")

    # Check against blocked libraries
    if normalized_name in BLOCKED_LIBRARIES:
        logging.warning(f"Blocked dangerous library: {library_name}")
        return False

    # Check for partial matches in blocked libraries
    for blocked in BLOCKED_LIBRARIES:
        if blocked in normalized_name or normalized_name in blocked:
            logging.warning(f"Blocked library due to partial match: {library_name}")
            return False

    return True


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
    # Validate all packages for security before installation
    for package in packages:
        if not validate_library_security(package):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Security Violation",
                    "details": f"Library '{package}' is not allowed for security reasons.",
                    "debug": "TEST MESSAGE"
                },
            )

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        DEPENDENCY_PATH,
        "--no-cache-dir",  # Prevent cache poisoning
        "--disable-pip-version-check",  # Reduce network calls
        "--no-warn-script-location",  # Reduce noise
        *packages,
    ]
    logging.info(
        "Attempt [%d] - Executing pip install command: %s", attempt, " ".join(command)
    )

    # We'll open a subprocess asynchronously
    try:
        # Create the subprocess
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Wait up to 300 seconds for the pip install to complete
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            # kill the process to avoid orphaned children
            process.kill()
            await process.communicate()
            logging.error(
                "Timeout during pip install execution on attempt [%d].",
                attempt,
                exc_info=True,
            )
            raise HTTPException(
                status_code=408,
                detail={
                    "error": "Dependency Installation Timeout",
                    "details": "Installation timed out. Check server logs for further diagnostics.",
                    "debug": "TEST MESSAGE"
                },
            )

        # Convert bytes to string
        out_str = stdout.decode().strip()
        err_str = stderr.decode().strip()

        logging.debug("pip install stdout (attempt [%d]):\n%s", attempt, out_str)
        logging.debug(
            "pip install stderr (attempt [%d]):\n%s", attempt, err_str or "(none)"
        )

        # If the return code is non-zero, we might need to retry or fail.
        if process.returncode != 0:
            logging.error(
                "pip install command failed (attempt [%d]). Return Code: %s | Stdout: %s | Stderr: %s",
                attempt,
                process.returncode,
                out_str,
                err_str,
                exc_info=True,
            )
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=command, output=stdout, stderr=stderr
            )

    except HTTPException:
        # Just re-raise any HTTPException for handling in outer scope
        raise
    except subprocess.CalledProcessError as cpe:
        # If we haven't exceeded max retries, we can retry
        if attempt < MAX_INSTALL_RETRIES:
            logging.warning(
                "Retrying pip install (attempt [%d -> %d]) for packages: %s",
                attempt,
                attempt + 1,
                packages,
            )
            await run_pip_install_async(packages, attempt=attempt + 1)
        else:
            logging.error("Max install retries reached. Failing out.")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Dependency Installation Error",
                    "details": "Repeated pip errors during installation. Check server logs for specifics.",
                    "debug": "TEST MESSAGE"
                },
            )
    except Exception as exc:
        tb = traceback.format_exc()
        logging.critical(
            "Unexpected error in run_pip_install_async (attempt [%d]): %s", attempt, tb
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Unexpected Dependency Installation Error",
                "details": "An unexpected error occurred during installation. Check server logs for traceback.",
                "debug": "TEST MESSAGE",
            },
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
    logging.debug(
        "START: Async dependency installation process. Input: %s", dependencies
    )

    # Validate input strictly
    if not isinstance(dependencies, list) or not all(
        isinstance(dep, str) and dep.strip() for dep in dependencies
    ):
        logging.error(
            "Invalid dependencies input. Must be a list of non-empty strings."
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid Dependency Input",
                "details": "Dependencies must be a list of non-empty strings.",
                "debug": "TEST MESSAGE"
            },
        )

    if not dependencies:
        logging.warning("No dependencies provided; skipping installation.")
        return

    # Ensure the dependency path is valid
    if not os.path.isdir(DEPENDENCY_PATH):
        logging.error(
            "Dependency path '%s' is not a valid directory or does not exist.",
            DEPENDENCY_PATH,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid Dependency Path",
                "details": f"Configured dependency path '{DEPENDENCY_PATH}' is invalid. Check server config.",
                "debug": "TEST MESSAGE"
            },
        )

    global _installed_packages
    # Acquire a thread-level lock for the global cache initialization
    try:
        async with asyncio.Lock():
            if _installed_packages is None:
                logging.debug(
                    "Initializing the global installed packages cache asynchronously."
                )
                _installed_packages = await get_installed_packages_async()
    except Exception as exc:
        logging.error("Cache initialization error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Cache Initialization Error",
                "details": "Error initializing package cache. Check server logs.",
                "debug": "TEST MESSAGE"
            },
        )

    # Compute which dependencies are missing (case-insensitive)
    lower_installed = _installed_packages  # already lowercased
    missing = [dep for dep in dependencies if dep.lower() not in lower_installed]

    if not missing:
        logging.info("All dependencies are already installed; nothing to do.")
        return

    logging.info("Missing dependencies: %s", missing)

    # Install the missing dependencies asynchronously (with retries if needed).
    await run_pip_install_async(missing)

    # Once installed, update the global cache again in a thread-safe manner
    async with asyncio.Lock():
        _installed_packages.update(dep.lower() for dep in missing)

    # Ensure that the newly installed packages are in sys.path
    if DEPENDENCY_PATH not in sys.path:
        sys.path.append(DEPENDENCY_PATH)
        logging.debug("Appended dependency path '%s' to sys.path.", DEPENDENCY_PATH)

    logging.info("Successfully installed dependencies: %s", missing)


def prepare_user_code(
    code: str, input_vars: Optional[Dict] = None, output_vars: Optional[List] = None
) -> str:
    # Inject input variables from the dictionary into the user's code
    input_vars_code = "\n".join(
        [f"{key} = {repr(value)}" for key, value in (input_vars or {}).items()]
    )
    output_vars_condition = (
        f"and not (k in {str(list((input_vars or {}).keys()))})"
        if not output_vars
        else f" and k in {str(output_vars)}"
    )

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


def normalize_code(s: str) -> str:
    """
    Unwraps if the entire payload is a single quoted string (e.g. "import os\\n..."),
    then converts literal backslash-newlines to real newlines.
    """
    t = s.strip()

    if (
        t.startswith(("'", '"', 'r"', "r'", 'u"', "u'", 'b"', "b'"))
        and t.endswith(t[0])
        and "\n" not in t[:1]
    ):
        try:
            t = ast.literal_eval(t)
        except Exception:
            pass

    t = t.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")

    return t


def extract_imports(code: str):
    """
    Parse Python code and extract imported top-level modules.
    Works for `import ...` and `from ... import ...`.
    """
    imports = set()
    code = normalize_code(code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports  # invalid code

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports


async def execute_python_code(
    code: str,
    input_vars: Optional[Dict[str, Any]] = None,
    output_vars: Optional[List[str]] = None,
) -> Dict[str, Any]:
    try:
        # Add resource limits
        resource_limit_code = f"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, ({MAX_EXECUTION_TIME}, {MAX_EXECUTION_TIME}))
resource.setrlimit(resource.RLIMIT_AS, ({MAX_MEMORY}, {MAX_MEMORY}))
"""
        full_code = (
            resource_limit_code
            + "\n"
            + prepare_user_code(code, input_vars, output_vars)
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
            tmp_file.write(full_code.encode("utf-8"))
            tmp_file_name = tmp_file.name

        try:
            # Execute as subprocess
            result = subprocess.run(
                ["python3", tmp_file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=MAX_EXECUTION_TIME,
                text=True,
            )

            # Clean up
            os.remove(tmp_file_name)

            stripped_stdout = result.stdout.strip() if result.stdout else None
            stripped_stderr = result.stderr.strip() if result.stderr else None

            if result.returncode == 0 and stripped_stdout:
                return {"result": json.loads(stripped_stdout), "error": None}
            else:
                try:
                    error_dict = json.loads(stripped_stderr)
                except Exception:
                    error_dict = None
                return {
                    "result": None,
                    "error": "Execution Error",
                    "details": error_dict["error"] if error_dict else stripped_stderr,
                }

        except subprocess.TimeoutExpired:
            os.remove(tmp_file_name)
            return {
                "result": None,
                "error": "Execution Error",
                "details": "Execution time limit exceeded",
            }

    except Exception as e:
        return {
            "result": None,
            "error": "Execution Error",
            "details": traceback.format_exc(),
            "code": code,
            "debug": "TEST MESSAGE",
        }


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    """Execute Python code with optional dependencies and input/output variables."""
    if not request.code:
        raise HTTPException(status_code=400, detail="No code provided")

    # Install missing dependencies
    if request.dependencies:
        await install_dependencies_async(request.dependencies)

    # Validate imports
    imported_libs = extract_imports(request.code)
    for lib in imported_libs:
        if not validate_library_security(lib):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Security Violation",
                    "details": f"Library '{lib}' is not allowed for security reasons.",
                    "debug": "TEST MESSAGE",
                },
            )

    # Execute user-provided Python code
    result = await execute_python_code(
        request.code, request.input_vars, request.output_vars
    )

    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": result["error"],
                "details": result["details"],
                "debug": "TEST MESSAGE",
            },
        )

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
