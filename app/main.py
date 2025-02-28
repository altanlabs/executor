import json
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import subprocess
import traceback
from typing import Dict, List, Optional, Any
import pkg_resources
import site

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

# Persistent directory for installed dependencies
DEPENDENCY_PATH = "/mnt/deps"

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

def execute_python_code(
    code: str,
    input_vars: Optional[Dict[str, Any]] = None,
    output_vars: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Execute Python code dynamically and return the results."""
    exec_globals = {}
    exec_locals = {}

    # Inject input variables
    if input_vars:
        exec_globals.update(input_vars)

    try:
        with io.StringIO() as output_buffer, io.StringIO() as error_buffer:
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, exec_globals, exec_locals)

                # Extract required output variables
                if output_vars:
                    vars_dict = {k: v for k, v in exec_locals.items() if k in output_vars}
                else:
                    vars_dict = {k: v for k, v in exec_locals.items() 
                                if not k.startswith('__') and not callable(v)}
                    if input_vars:
                        vars_dict = {k: v for k, v in vars_dict.items() 
                                   if k not in input_vars}

                return {
                    "result": {
                        "vars": vars_dict,
                        "debug": output_buffer.getvalue()
                    },
                    "error": None
                }
    except Exception as e:
        return {
            "result": None,
            "error": "Execution Error",
            "details": traceback.format_exc()
        }

@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    """Execute Python code with optional dependencies and input/output variables."""
    if not request.code:
        raise HTTPException(status_code=400, detail="No code provided")

    # Install missing dependencies
    try:
        install_dependencies(request.dependencies)
    except Exception as e:
        return {
            "result": None,
            "error": "Dependency Error",
            "details": str(e)
        }

    # Execute user-provided Python code
    result = execute_python_code(
        request.code,
        request.input_vars,
        request.output_vars
    )

    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": result["error"],
                "details": result["details"]
            }
        )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 