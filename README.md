# Python Code Executor API

A FastAPI-based service that allows dynamic Python code execution with dependency management, designed for AWS Fargate deployment.

## Features

- Dynamic Python code execution
- On-demand dependency installation
- Input/output variable management
- Stdout/stderr capture
- AWS Fargate ready

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Docker Build and Run

1. Build the Docker image:
```bash
docker build -t python-executor .
```

2. Run the container:
```bash
docker run -p 8080:8080 python-executor
```

## AWS Fargate Deployment

1. Create an ECR repository
2. Build and tag the Docker image
3. Push to ECR
4. Create a Fargate task definition
5. Deploy as a Fargate service

## API Usage

### Execute Code Endpoint

`POST /execute`

Request body:
```json
{
    "code": "print('Hello, World!')\nresult = 42",
    "input_vars": {
        "x": 10,
        "y": 20
    },
    "output_vars": ["result"],
    "dependencies": ["numpy", "pandas"]
}
```

Response:
```json
{
    "result": "{\"result\": 42}",
    "debug": "Hello, World!\n",
    "error": null
}
```

## Security Considerations

- The API executes arbitrary Python code, so it should be deployed in a secure environment
- Consider implementing authentication and authorization
- Use appropriate IAM roles and security groups in AWS
- Monitor resource usage to prevent abuse

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc` 