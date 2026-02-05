#!/usr/bin/env bash

docker build -t uom-neighbor-selection \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) .
