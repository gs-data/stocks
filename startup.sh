#!/bin/bash

docker run -dp 27017:27017 \
  --name mongodb \
  --mount type=bind,source="$(pwd)"/data/mongodb,target=/data/db \
  mongo