#!/bin/bash
# Wrapper script to run chat-manager.py in the bff conda environment
set -e  # Exit on error

# Set up environment
export HOME=/home/cohab
export USER=cohab

# Initialize conda properly for non-interactive shells
if [ -f /home/cohab/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/cohab/miniconda3/etc/profile.d/conda.sh
else
    echo "Error: conda.sh not found" >&2
    exit 1
fi

# Activate the bff environment
conda activate bff

# Explicitly use Python 3.10 (the system Python that conda uses)
# This ensures we use the same Python version that has numpy installed
PYTHON_CMD="/usr/bin/python3"

# Verify Python version matches what we expect
PYTHON_VERSION=$("$PYTHON_CMD" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "Error: Expected Python 3.10, but got $PYTHON_VERSION" >&2
    echo "Python path: $PYTHON_CMD" >&2
    exit 1
fi

# Ensure Python can find packages in user's local site-packages
export PYTHONPATH="${HOME}/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"

# Change to the script directory
cd "$(dirname "$0")"

# Run the chat manager using the explicit Python 3.10
# Use exec to replace the shell process
exec "$PYTHON_CMD" chat-manager.py "$@"
