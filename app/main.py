import json
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
import subprocess
import traceback
from typing import Dict, List, Optional, Any

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
DEPENDENCY_PATH = "/tmp/deps"

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
    result: Optional[str] = None
    debug: Optional[str] = None
    error: Optional[str] = None

def install_dependencies(dependencies: List[str]) -> None:
    """Install Python dependencies if they're not already installed."""
    if not dependencies:
        return
    
    try:
        print(f"Installing dependencies: {dependencies}")  # Debug log
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", DEPENDENCY_PATH, *dependencies],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Installation output: {result.stdout}")  # Debug log
        print(f"Installation errors: {result.stderr}")  # Debug log
        
        if DEPENDENCY_PATH not in sys.path:
            sys.path.append(DEPENDENCY_PATH)
            print(f"Added {DEPENDENCY_PATH} to sys.path")  # Debug log
            
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to install dependencies: {e.stderr}"
        )

def execute_python_code(
    code: str,
    input_vars: Optional[Dict[str, Any]] = None,
    output_vars: Optional[List[str]] = None
) -> Dict[str, str]:
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

                # Extract only required output variables
                output_vars_dict = {k: v for k, v in exec_locals.items()
                                  if not output_vars or k in output_vars}

                return {
                    "result": json.dumps(output_vars_dict),
                    "debug": output_buffer.getvalue(),
                    "error": error_buffer.getvalue()
                }
    except Exception as e:
        return {"error": traceback.format_exc()}

@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    """
    Execute Python code with optional dependencies and input/output variables.
    
    - code: The Python code to execute
    - input_vars: Dictionary of variables to inject into the execution context
    - output_vars: List of variable names to extract from the execution context
    - dependencies: List of Python packages to install before execution
    """
    if not request.code:
        raise HTTPException(status_code=400, detail="No code provided")

    # Install missing dependencies
    try:
        install_dependencies(request.dependencies)
    except Exception as e:
        return ExecuteResponse(error=f"Failed to install dependencies: {str(e)}")

    # Execute user-provided Python code
    result = execute_python_code(
        request.code,
        request.input_vars,
        request.output_vars
    )

    return ExecuteResponse(**result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 