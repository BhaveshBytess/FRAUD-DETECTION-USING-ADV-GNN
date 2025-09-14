@echo off
REM Windows deployment script for hHGTN Demo Service

setlocal EnableDelayedExpansion

echo ^🚀 hHGTN Demo Service Deployment Script (Windows)
echo ================================================

REM Configuration
set SERVICE_NAME=hhgtn-demo
set IMAGE_NAME=hhgtn-demo-service
if "%TAG%"=="" set TAG=latest
if "%ENV%"=="" set ENV=production

REM Check if Docker is installed and running
echo ^ℹ️  Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ^❌ Docker is not installed. Please install Docker first.
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo ^❌ Docker is not running. Please start Docker first.
    exit /b 1
)
echo ^✅ Docker is available

REM Check if Docker Compose is available
echo ^ℹ️  Checking Docker Compose...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ^❌ Docker Compose is not available. Please install Docker Compose.
        exit /b 1
    )
)
echo ^✅ Docker Compose is available

REM Handle command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=deploy

if "%COMMAND%"=="build" goto build
if "%COMMAND%"=="deploy" goto deploy
if "%COMMAND%"=="start" goto start
if "%COMMAND%"=="stop" goto stop
if "%COMMAND%"=="restart" goto restart
if "%COMMAND%"=="status" goto status
if "%COMMAND%"=="logs" goto logs
if "%COMMAND%"=="test" goto test
if "%COMMAND%"=="help" goto help
goto help

:build
echo ^ℹ️  Building Docker image: %IMAGE_NAME%:%TAG%
docker build --target %ENV% --tag %IMAGE_NAME%:%TAG% --tag %IMAGE_NAME%:latest .
if errorlevel 1 (
    echo ^❌ Docker build failed
    exit /b 1
)
echo ^✅ Docker image built successfully
goto end

:deploy
call :build
if errorlevel 1 exit /b 1
call :start
if errorlevel 1 exit /b 1
call :check_health
call :show_status
goto end

:start
echo ^ℹ️  Deploying with Docker Compose (%ENV% mode)...
if "%ENV%"=="development" (
    docker-compose --profile dev up -d
) else (
    docker-compose up -d
)
if errorlevel 1 (
    echo ^❌ Docker Compose deployment failed
    exit /b 1
)
echo ^✅ Service deployed successfully
goto end

:stop
echo ^ℹ️  Stopping and removing containers...
docker-compose down
echo ^✅ Cleanup completed
goto end

:restart
call :stop
call :start
call :check_health
call :show_status
goto end

:status
call :show_status
goto end

:logs
docker-compose logs -f demo-service
goto end

:test
echo ^ℹ️  Running API test...
curl -X POST "http://localhost:8000/predict" ^
    -H "Content-Type: application/json" ^
    -d "{\"transaction\": {\"user_id\": \"test_user\", \"merchant_id\": \"test_merchant\", \"device_id\": \"test_device\", \"ip_address\": \"192.168.1.100\", \"timestamp\": \"2024-01-15T10:30:00Z\", \"amount\": 100.0, \"currency\": \"USD\"}, \"explain_config\": {\"top_k_nodes\": 10, \"top_k_edges\": 15}}"
goto end

:check_health
echo ^ℹ️  Checking service health...
timeout /t 10 /nobreak >nul

set /a attempt=1
set /a max_attempts=30

:health_loop
curl -sf http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    echo ^✅ Service is healthy and responding
    goto :eof
)

echo ^ℹ️  Attempt !attempt!/%max_attempts%: Service not ready yet...
timeout /t 2 /nobreak >nul
set /a attempt+=1
if !attempt! leq %max_attempts% goto health_loop

echo ^❌ Service health check failed after %max_attempts% attempts
exit /b 1

:show_status
echo ^ℹ️  Service status:
docker-compose ps
echo.
echo ^ℹ️  Service logs (last 10 lines):
docker-compose logs --tail=10 demo-service
echo.
echo ^ℹ️  Service endpoints:
echo   ^📊 Health Check: http://localhost:8000/health
echo   ^📚 API Docs: http://localhost:8000/docs  
echo   ^🌐 Demo UI: http://localhost:8000
echo   ^📈 Metrics: http://localhost:8000/metrics
goto :eof

:help
echo Usage: %0 {build^|deploy^|start^|stop^|restart^|status^|logs^|test^|help}
echo.
echo Commands:
echo   build    - Build Docker image only
echo   deploy   - Full deployment (build + start + health check)
echo   start    - Start services using existing images
echo   stop     - Stop and remove all containers
echo   restart  - Stop and restart services
echo   status   - Show service status and logs
echo   logs     - Follow service logs
echo   test     - Run API test call
echo   help     - Show this help message
echo.
echo Environment variables:
echo   ENV      - deployment environment (production^|development) [default: production]
echo   TAG      - Docker image tag [default: latest]
goto end

:end
endlocal
