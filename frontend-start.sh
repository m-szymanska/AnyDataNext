#!/bin/bash

# Colors for output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Backend URL to check (adjust if needed)
BACKEND_CHECK_URL="http://localhost:8000"
# Simple endpoint that should exist, like docs or root
BACKEND_HEALTH_ENDPOINT="/docs" 

# Frontend directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo -e "${YELLOW}--- Starting AnyDataset Frontend ---${NC}"

echo "Checking if backend is responding at ${BACKEND_CHECK_URL}${BACKEND_HEALTH_ENDPOINT}..."

# Use curl to check if the backend is accessible
# -s: Silent mode (no progress meter)
# -o /dev/null: Discard output
# -w '%{http_code}': Output only the HTTP status code
# --connect-timeout 5: Max time in seconds for connection
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${BACKEND_CHECK_URL}${BACKEND_HEALTH_ENDPOINT}")

# Check the HTTP status code
if [ "$HTTP_STATUS" -ge 200 ] && [ "$HTTP_STATUS" -lt 400 ]; then
    echo -e "${GREEN}Backend is responding (HTTP Status: $HTTP_STATUS). Proceeding to start frontend.${NC}"
else
    echo -e "${RED}Error: Backend is not responding or returned an error (HTTP Status: $HTTP_STATUS).${NC}"
    echo -e "${RED}Please ensure the backend is running correctly (using ./backend-start.sh in another terminal).${NC}"
    exit 1
fi

# Navigate to the frontend directory
cd "$FRONTEND_DIR" || {
    echo "Error: Could not navigate to frontend directory: $FRONTEND_DIR"
    exit 1
}

echo "Current directory: $(pwd)"

# Check for node_modules
if [ ! -d "node_modules" ]; then
    echo "Warning: node_modules directory not found. Running 'npm install'..."
    npm install || {
        echo -e "${RED}Error: Failed to install frontend dependencies.${NC}"
        exit 1
    }
    echo -e "${GREEN}Frontend dependencies installed successfully.${NC}"
fi


echo -e "${YELLOW}Starting frontend development server (npm run dev)...${NC}"
echo -e "${GREEN}Frontend zaraz wystartuje. Jak zobaczysz logi Next.js, a w przeglądarce pojawi się strona, to znaczy, że jest cycuś malina! ;) ${NC}\n"

# Run the frontend development server
npm run dev

# echo "Frontend development server stopped."
