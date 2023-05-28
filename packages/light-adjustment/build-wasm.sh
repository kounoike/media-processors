#! /bin/bash

set -eux

cd zig/
zig build -Doptimize=ReleaseFast -Dtarget=wasm32-freestanding -Dcpu=bleeding_edge
cd ..

mkdir -p dist/
cp zig/zig-out/lib/light_adjustment.wasm dist/
