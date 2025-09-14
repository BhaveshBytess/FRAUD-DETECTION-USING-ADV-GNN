#!/bin/bash
# Deployment script for hHGTN Demo Service

set -e

echo "🚀 hHGTN Demo Service Deployment Script"
echo "========================================"

# Configuration
SERVICE_NAME="hhgtn-demo"
IMAGE_NAME="hhgtn-demo-service"
TAG="${TAG:-latest}"
ENV="${ENV:-production}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if Docker is installed and running
check_docker() {
    info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
    fi
    
    success "Docker is available"
}

# Check if Docker Compose is available
check_docker_compose() {
    info "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not available. Please install Docker Compose."
    fi
    success "Docker Compose is available"
}

# Build Docker image
build_image() {
    info "Building Docker image: ${IMAGE_NAME}:${TAG}"
    docker build \
        --target ${ENV} \
        --tag ${IMAGE_NAME}:${TAG} \
        --tag ${IMAGE_NAME}:latest \
        .
    success "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_compose() {
    info "Deploying with Docker Compose (${ENV} mode)..."
    
    if [ "${ENV}" = "development" ]; then
        docker-compose --profile dev up -d
    else
        docker-compose up -d
    fi
    
    success "Service deployed successfully"
}

# Check service health
check_health() {
    info "Checking service health..."
    
    # Wait for service to start
    sleep 10
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            success "Service is healthy and responding"
            return 0
        fi
        
        info "Attempt $attempt/$max_attempts: Service not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    error "Service health check failed after $max_attempts attempts"
}

# Show service status
show_status() {
    info "Service status:"
    docker-compose ps
    
    echo ""
    info "Service logs (last 10 lines):"
    docker-compose logs --tail=10 demo-service
    
    echo ""
    info "Service endpoints:"
    echo "  📊 Health Check: http://localhost:8000/health"
    echo "  📚 API Docs: http://localhost:8000/docs"
    echo "  🌐 Demo UI: http://localhost:8000"
    echo "  📈 Metrics: http://localhost:8000/metrics"
}

# Cleanup function
cleanup() {
    info "Stopping and removing containers..."
    docker-compose down
    success "Cleanup completed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "build")
            check_docker
            build_image
            ;;
        "deploy")
            check_docker
            check_docker_compose
            build_image
            deploy_compose
            check_health
            show_status
            ;;
        "start")
            check_docker_compose
            deploy_compose
            check_health
            show_status
            ;;
        "stop")
            cleanup
            ;;
        "restart")
            cleanup
            deploy_compose
            check_health
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            docker-compose logs -f demo-service
            ;;
        "test")
            info "Running API test..."
            curl -X POST "http://localhost:8000/predict" \
                -H "Content-Type: application/json" \
                -d '{
                    "transaction": {
                        "user_id": "test_user",
                        "merchant_id": "test_merchant", 
                        "device_id": "test_device",
                        "ip_address": "192.168.1.100",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "amount": 100.0,
                        "currency": "USD"
                    },
                    "explain_config": {
                        "top_k_nodes": 10,
                        "top_k_edges": 15
                    }
                }' | python -m json.tool
            ;;
        "help"|*)
            echo "Usage: $0 {build|deploy|start|stop|restart|status|logs|test|help}"
            echo ""
            echo "Commands:"
            echo "  build    - Build Docker image only"
            echo "  deploy   - Full deployment (build + start + health check)"
            echo "  start    - Start services using existing images"
            echo "  stop     - Stop and remove all containers"
            echo "  restart  - Stop and restart services"
            echo "  status   - Show service status and logs"
            echo "  logs     - Follow service logs"
            echo "  test     - Run API test call"
            echo "  help     - Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ENV      - deployment environment (production|development) [default: production]"
            echo "  TAG      - Docker image tag [default: latest]"
            exit 0
            ;;
    esac
}

# Run main function with all arguments
main "$@"
