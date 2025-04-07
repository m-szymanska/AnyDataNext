#!/bin/bash
# frontend_init.sh - Initializes the frontend for VistaCare Node.js application.
# This script:
#   1. Checks prerequisites (node, pnpm)
#   2. Checks backend health endpoint
#   3. Sets up environment files
#   4. Installs dependencies
#   5. Starts development server

# Note: Requires Node.js >=14.0.0 as per package.json

# ---------------------
# Color settings for messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------------------
# Message functions
error_exit() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

warn_msg() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

success_msg() {
    echo -e "${GREEN}✅ $1${NC}"
}

# ---------------------
# Function to check if a command exists
check_command() {
    command -v "$1" >/dev/null 2>&1 || error_exit "$1 is required but not installed."
}

# ---------------------
# Check backend health endpoint
check_backend() {
    echo "🔍 Checking if backend is running..."
    if ! curl -s "http://localhost:3001/health" > /dev/null; then
        warn_msg "Backend is not running. Please start the server first!"
        read -p "Do you want me to try starting the server? [Y/n]: " start_server
        if [[ ! "$start_server" =~ ^[Nn]$ ]]; then
            cd server || error_exit "Cannot find server directory"
            bash server_init.sh &
            cd ..
            echo "Waiting for server to start..."
            sleep 5
            if ! curl -s "http://localhost:3001/health" > /dev/null; then
                error_exit "Server failed to start. Please check server logs."
            fi
        else
            error_exit "Please start the server first using server_init.sh"
        fi
    fi
    success_msg "Backend is running"
}

# ---------------------
# Setup environment files
setup_env() {
    if [ ! -f .env ]; then
        echo "🔍 No .env file found, copying from .env.development..."
        cp .env.development .env || error_exit "Failed to copy .env.development to .env"
        success_msg "Environment file created from .env.development"
    else
        success_msg "Environment file (.env) exists"
    fi
}

# ---------------------
# Main execution
echo "🔍 Checking prerequisites..."
check_command node
check_command pnpm

# Check if backend is running
check_backend

# Setup environment files
setup_env

echo "🚀 Installing frontend dependencies..."
pnpm install || error_exit "Failed to install frontend dependencies"
success_msg "Frontend dependencies installed"

echo "🌟 Starting frontend in development mode..."
pnpm dev || error_exit "Failed to start frontend"