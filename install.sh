#!/bin/bash
set -e
cd "$(dirname "$0")"
# Check for Python and exit if not found
if ! [ -x "$(command -v python)" ]; then
  echo 'Error: python is not installed.' >&2
  exit 1
fi
# Create a virtual environment
if [ ! -d "venv" ]; then
  python -m venv venv
fi

# Upgrade pip before `pip install`
python -m pip install --upgrade pip

# Install the custom diffusers version from GitHub
./venv/bin/pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
./venv/bin/pip install -r requirements.txt

echo "Installation completed."
