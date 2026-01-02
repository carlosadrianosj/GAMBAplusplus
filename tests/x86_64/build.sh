#!/bin/bash
# Build script for x86_64 test binary

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="gamba-test-x86_64"
CONTAINER_NAME="gamba-test-x86_64-build"

echo "Building x86_64 test binary..."

# Build Docker image
docker build -t $IMAGE_NAME .

# Run container to compile
docker run --name $CONTAINER_NAME $IMAGE_NAME

# Extract artifacts
docker cp $CONTAINER_NAME:/build/test_mba.o $SCRIPT_DIR/
docker cp $CONTAINER_NAME:/build/test_mba.asm $SCRIPT_DIR/
docker cp $CONTAINER_NAME:/build/test_mba.symbols $SCRIPT_DIR/

# Cleanup
docker rm $CONTAINER_NAME

echo "Build complete. Artifacts:"
echo "  - test_mba.o"
echo "  - test_mba.asm"
echo "  - test_mba.symbols"

