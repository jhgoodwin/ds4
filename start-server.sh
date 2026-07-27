#!/bin/bash
# Start ds4-server with DeepSeek V4 Flash at 256K context, no MTP/DSPARK.
# High thinking (default at <384K ctx), port 11436, all binds

export DS4_THREADS=${DS4_THREADS:-24}

exec ./ds4-server \
    --gpu-devices 0,1 \
    --gpu-vram 92,92 \
    --ctx 262144 \
    --host 0.0.0.0 \
    --port 11436 \
    --warm-weights \
    "$@"
