import json
import sys
import os
import io
import traceback
import tempfile
import asyncio
import subprocess
from typing import Dict, List, Optional, Any
import pkg_resources
import site

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Constants
MAX_EXECUTION_TIME = 300  # seconds
MAX_MEMORY = 256 * 1024 * 1024  # 256MB
DEPENDENCY_PATH = "/mnt/deps"

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

def get_installed_packages() -> set:
    """Get a set of all installed packages across all paths."""
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    
    # Add packages from our custom path if it exists
    if os.path.exists(DEPENDENCY_PATH):
        site.addsitedir(DEPENDENCY_PATH)
        installed_packages.update(
            {pkg.key for pkg in pkg_resources.find_distributions(DEPENDENCY_PATH)}
        )
    return installed_packages

# Cache for installed packages
_installed_packages = None

def install_dependencies(dependencies: List[str]) -> None:
    """Install Python dependencies if they're not already installed."""
    if not dependencies:
        return
        
    global _installed_packages
    
    # Initialize cache if needed
    if _installed_packages is None:
        _installed_packages = get_installed_packages()
    
    # Normalize package names (convert to lowercase)
    missing_deps = [
        dep for dep in dependencies 
        if dep.lower() not in _installed_packages
    ]
    
    if not missing_deps:
        print("All dependencies already installed")
        return
        
    try:
        print(f"Installing missing dependencies: {missing_deps}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", DEPENDENCY_PATH, *missing_deps],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Update the cache with new packages
        _installed_packages.update(pkg.lower() for pkg in missing_deps)
        
        if DEPENDENCY_PATH not in sys.path:
            sys.path.append(DEPENDENCY_PATH)
            
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to install dependencies: {e.stderr}"
        )

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
            install_dependencies(request.dependencies)
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