#!/bin/bash



set -e  # Exit on error

echo "=========================================="
echo "Spam Detection System - Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if Python is installed
echo "Step 1: Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
print_success "Python $PYTHON_VERSION found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi
print_success "pip3 found"

# Create project structure
echo ""
echo "Step 2: Creating project structure..."

mkdir -p app
mkdir -p models
mkdir -p train
mkdir -p frontend

print_success "Directory structure created"

# Create __init__.py for app package
touch app/__init__.py
print_success "Python package initialized"

# Install training dependencies
echo ""
echo "Step 3: Installing dependencies for model training..."
print_info "This may take a few minutes..."

pip3 install pandas scikit-learn joblib numpy --quiet

print_success "Training dependencies installed"

# Check if model exists
echo ""
echo "Step 4: Checking for trained model..."

if [ -f "models/spam_pipeline.joblib" ]; then
    print_success "Model found at models/spam_pipeline.joblib"
    read -p "Do you want to retrain the model? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TRAIN_MODEL=true
    else
        TRAIN_MODEL=false
    fi
else
    print_info "No trained model found. Training new model..."
    TRAIN_MODEL=true
fi

# Train model if needed
if [ "$TRAIN_MODEL" = true ]; then
    echo ""
    echo "Step 5: Training spam detection model..."
    
    if [ -f "train/spam_train.py" ]; then
        cd train
        python3 spam_train.py
        cd ..
        print_success "Model training completed"
    else
        print_error "Training script not found at train/spam_train.py"
        print_info "Please create the training script first"
        exit 1
    fi
else
    print_info "Skipping model training"
fi

# Install API dependencies
echo ""
echo "Step 6: Installing API dependencies..."

if [ -f "app/requirements.txt" ]; then
    pip3 install -r app/requirements.txt --quiet
    print_success "API dependencies installed"
else
    print_error "requirements.txt not found at app/requirements.txt"
    exit 1
fi

# Verify all required files exist
echo ""
echo "Step 7: Verifying project files..."

FILES=(
    "app/main.py"
    "app/model.py"
    "app/requirements.txt"
    "frontend/index.html"
    "Dockerfile"
    "models/spam_pipeline.joblib"
)

MISSING_FILES=()

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_error "$file is missing"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    print_error "Some required files are missing. Please create them before proceeding."
    exit 1
fi

# Check if Docker is installed (optional)
echo ""
echo "Step 8: Checking Docker (optional)..."

if command -v docker &> /dev/null; then
    print_success "Docker found"
    
    read -p "Do you want to build Docker image? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Building Docker image..."
        docker build -t spam-detector .
        print_success "Docker image built successfully"
    fi
else
    print_info "Docker not found (optional - for containerized deployment)"
fi

# Initialize git repository (if not exists)
echo ""
echo "Step 9: Setting up version control..."

if [ -d ".git" ]; then
    print_success "Git repository already initialized"
else
    git init
    print_success "Git repository initialized"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Environment
.env

# Logs
*.log
EOF
    print_success ".gitignore created"
fi

# Test the API
echo ""
echo "Step 10: Starting test server..."
print_info "Starting server on http://localhost:8080"
print_info "Press Ctrl+C to stop the server after testing"

# Start server in background
uvicorn app.main:app --host 0.0.0.0 --port 8080 &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Test health endpoint
echo ""
echo "Testing API endpoints..."

if curl -s http://localhost:8080/health > /dev/null; then
    print_success "Server is running and healthy"
    
    # Test prediction endpoint
    TEST_RESPONSE=$(curl -s -X POST http://localhost:8080/predict_spam \
        -H "Content-Type: application/json" \
        -d '{"text":"You have won a free prize!"}')
    
    if echo "$TEST_RESPONSE" | grep -q "is_spam"; then
        print_success "Prediction endpoint working"
        echo "Test prediction result: $TEST_RESPONSE" | python3 -m json.tool
    else
        print_error "Prediction endpoint test failed"
    fi
else
    print_error "Server health check failed"
fi

# Stop server
kill $SERVER_PID 2>/dev/null || true

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Project structure:"
echo "  ✓ app/             - FastAPI application"
echo "  ✓ models/          - Trained ML model"
echo "  ✓ train/           - Training scripts"
echo "  ✓ frontend/        - Web dashboard"
echo ""
echo "Next steps:"
echo "  1. Start the server:"
echo "     uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"
echo ""
echo "  2. Open your browser:"
echo "     http://localhost:8080"
echo ""
echo "  3. View API docs:"
echo "     http://localhost:8080/docs"
echo ""
echo "  4. Run tests:"
echo "     python3 test_api.py"
echo ""
echo "  5. Deploy to production:"
echo "     See DEPLOYMENT.md for detailed instructions"
echo ""
echo "For help: https://github.com/yourusername/spam-detector"
echo "=========================================="