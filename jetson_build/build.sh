#!/usr/bin/bash

set -e

mkdir -p jetsonfs
# sshfs me@192.168.50.198:/ jetsonfs -o allow_root
# docker build -v /tmp/jetsonfs/home/me:/home/me -v /tmp/jetsonfs/usr/local/cuda:/usr/local/cuda -v /tmp/jetsonfs/usr/lib/aarch64-linux-gnu:/hostlibs
docker build . -t quant/jetsonbuild --platform linux/arm64 --network=host
# docker run --runtime nvidia -it --rm --network=host -v /tmp/jetsonfs/home/me:/home/me -v /tmp/jetsonfs/usr/local/cuda:/usr/local/cuda -v /tmp/jetsonfs/usr/lib/aarch64-linux-gnu:/hostlibs ./Dockerfile
