#! /bin/bash

set -eux

cd zig/
zig build -Doptimize=ReleaseFast
cd ..

mkdir -p dist/
cp zig/zig-out/lib/light_adjustment.wasm dist/
