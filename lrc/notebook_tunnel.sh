#!/usr/bin/env bash

# Run on server:
# jupyter notebook --no-browser --port=8888

ssh -J sol2 kripner@tdll-8gpu1 -L 8888:localhost:8888 -N

