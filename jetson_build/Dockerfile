FROM arm64v8/ubuntu:22.04

# COPY jetsonfs/usr/local/cuda-11.8 /usr/local/cuda
# COPY jetsonfs/usr/lib/aarch64-linux-gnu /hostlibs

RUN apt-get update && apt-get -y install python3 cmake git build-essential wget

COPY nvidia-l4t-apt-source.list /etc/apt/sources.list.d/
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb && apt-get update && apt-get -y install cuda=11.8.0-1

# RUN cd /home/me/remotefs/onnxruntime && ./build.sh --config Release --update --build --parallel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /hostlibs --tensorrt_home /hostlibs

# export PATH="/cmake-3.31.11-linux-aarch64/bin/:${PATH}"
# export PATH="/home/me/remotefs/cmake-3.31.11-linux-aarch64/bin/:${PATH}"
# cd /home/me/remotefs/onnxruntime && useradd me && su me && ./build.sh --config Release --update --build --parallel --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /hostlibs --tensorrt_home /hostlibs
