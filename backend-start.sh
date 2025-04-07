#!/bin/bash

# Colors for output
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}--- Starting AnyDataset Backend ---${NC}"

# Navigate to the backend directory (assuming script is run from project root)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
BACKEND_DIR="$SCRIPT_DIR/backend"

cd "$BACKEND_DIR" || {
    echo -e "${RED}Error: Could not navigate to backend directory: $BACKEND_DIR${NC}"
    exit 1
}

echo "Current directory: $(pwd)"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Python virtual environment (.venv) not found in backend directory.${NC}"
    echo "Please create it first (e.g., python -m venv .venv or uv venv)"
    exit 1
fi

# Activate virtual environment
echo "Activating Python virtual environment..."
source .venv/bin/activate || {
    echo -e "${RED}Error: Failed to activate virtual environment.${NC}"
    exit 1
}

# Get Python version
PYTHON_VERSION=$(python --version)
echo "Using Python: $PYTHON_VERSION"

# Determine which package installer to use (Prioritize uv)
INSTALLER_CMD=""
INSTALLER_NAME=""

if command -v uv &> /dev/null; then
    echo -e "${GREEN}Found 'uv' command. Using it for dependency management.${NC}"
    INSTALLER_CMD="uv pip install -r requirements.txt -q"
    INSTALLER_NAME="uv"
elif command -v pip &> /dev/null; then
    echo -e "${YELLOW} 'uv' not found. Found 'pip' command. Using pip.${NC}"
    INSTALLER_CMD="pip install -r requirements.txt -q"
    INSTALLER_NAME="pip"
elif command -v python &> /dev/null && python -m pip --version &> /dev/null; then
    echo -e "${YELLOW}'uv' and 'pip' command not found directly. Found 'python -m pip'. Using fallback.${NC}"
    INSTALLER_CMD="python -m pip install -r requirements.txt -q"
    INSTALLER_NAME="python -m pip"
else
    echo -e "${RED}Error: Could not find a working package installer ('uv', 'pip', or 'python -m pip') even after activating .venv!${NC}"
    echo "This strongly suggests a corrupted virtual environment or critical PATH issues."
    echo "Please try recreating the virtual environment manually or ensure uv/pip is installed correctly."
    exit 1
fi


echo "Installing/updating backend dependencies from requirements.txt using '$INSTALLER_NAME'..."
$INSTALLER_CMD || {
    echo -e "${RED}Error: Failed to install dependencies using '$INSTALLER_NAME'.${NC}"
    exit 1
}

echo -e "${GREEN}Dependencies installed successfully.${NC}"

echo -e "${YELLOW}Starting Uvicorn server for the backend application...${NC}"
echo "Host: 0.0.0.0, Port: 8000, Reload enabled."
echo "Watching for changes in the current directory (.)"

echo -e "\n${GREEN}>>> Backend zaraz wystartuje. Jak zobaczysz logi Uvicorna, to znaczy, że u mnie wszystko cacy! ${NC}"
echo -e "${GREEN}>>> NIE ZAMYKAJ TEGO TERMINALA! ${NC}"
echo -e "${GREEN}>>> W drugim terminalu uruchom ${YELLOW}./frontend-start.sh${GREEN}, a będzie cycuś malina! ;) ${NC}\n"

# Run Uvicorn - this command will block the terminal until stopped (Ctrl+C)
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload --reload-dirs .

# This part will likely only execute after Uvicorn is stopped
# echo "Backend server stopped."
# Deactivate environment if Uvicorn stops gracefully (optional)
# deactivate
