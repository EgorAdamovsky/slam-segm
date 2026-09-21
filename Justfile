run:
 - cd vision_app && RUST_LOG=warn,vision_app=info cargo run

build_aarch:
 - cd vision_app && cross build --release --target=aarch64-unknown-linux-gnu


build_jetson_nano_deps:
 - cd jetson_build && ./build.sh