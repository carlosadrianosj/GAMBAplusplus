#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="gamba-test-mips32"
CONTAINER_NAME="gamba-test-mips32-build"

echo "Building MIPS32 test binary..."

docker build -t $IMAGE_NAME .

docker run --name $CONTAINER_NAME $IMAGE_NAME

docker cp $CONTAINER_NAME:/build/test_mba.o $SCRIPT_DIR/
docker cp $CONTAINER_NAME:/build/test_mba.asm $SCRIPT_DIR/
docker cp $CONTAINER_NAME:/build/test_mba.symbols $SCRIPT_DIR/

docker rm $CONTAINER_NAME

echo "Build complete."

