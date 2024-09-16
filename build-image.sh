#!/bin/bash

NOTE=$1
if [ -z "$NOTE" ]; then
    NOTE=official
fi
VERSION=$(python setup.py --version | tail -n1)
echo "Building image for version $VERSION with note $NOTE"

rm -rf dist
python setup.py sdist --formats=gztar
docker build \
    --build-arg VERSION=${VERSION} \
    -t xingyaoww/swe-bench-service:${NOTE}-${VERSION} \
    -f Dockerfile.cloud .

rm -rf dist

