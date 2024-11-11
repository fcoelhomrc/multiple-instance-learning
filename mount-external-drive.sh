#!/bin/bash

# Variables
REMOTE_USER="fcmoreira"
REMOTE_HOST="ctm-login.inesctec.pt"
LOCAL_MOUNT_POINT="$HOME/ExternalDrives/"

# Check if REMOTE_PATH argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <REMOTE_PATH>"
  exit 1
fi

# Use the argument as REMOTE_PATH
REMOTE_PATH="$1"

# Mount the remote drive
echo "Mounting remote drive..."
sshfs "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_MOUNT_POINT}"
if [ $? -ne 0 ]; then
  echo "Failed to mount remote drive. Make sure you are connected to VPN."
  exit 1
fi

