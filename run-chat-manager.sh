#!/bin/bash
# Wrapper script to run chat-manager.py with the right Python for this machine.
#
# The two Jetsons are built differently, so this script picks the interpreter
# rather than hardcoding one machine's layout:
#
#   venv/ present          -> helper.local: project-local venv, created with
#                             --system-site-packages plus .pth files bridging
#                             ~/.local (Jetson CUDA torch) and the unitree SDK
#                             source tree.
#   no venv/               -> snapper.local: system Python 3.10 with the CUDA
#                             stack in ~/.local/lib/python3.10/site-packages.
#
# Neither machine should need a locally-modified copy of this file. If you are
# about to edit it to make your machine work, add a branch here instead.
set -e  # Exit on error

# Change to the script directory
cd "$(dirname "$0")"

# CycloneDDS C library, needed by unitree_sdk2py. Only set it if the caller
# has not, and only if the build is actually there.
if [ -z "${CYCLONEDDS_HOME:-}" ] && [ -d "${HOME}/code/cyclonedds/install" ]; then
    export CYCLONEDDS_HOME="${HOME}/code/cyclonedds/install"
fi

# --- helper.local: project-local venv ---------------------------------------
if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    exec python chat-manager.py "$@"
fi

# --- snapper.local: system Python 3.10 + user site-packages -----------------
# Explicitly use Python 3.10; this is the interpreter that has numpy and the
# rest of the Jetson CUDA stack installed under ~/.local.
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

# Run the chat manager using the explicit Python 3.10
# Use exec to replace the shell process
exec "$PYTHON_CMD" chat-manager.py "$@"
