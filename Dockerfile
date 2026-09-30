FROM ubuntu:22.04
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
USER root

WORKDIR /workspace

RUN apt-get update && apt-get install -y openssh-client && apt-get install -y git
RUN mkdir -p /root/.ssh
COPY ./thurl /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa && \
ssh-keyscan github.com >> /root/.ssh/known_hosts

COPY flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl /workspace/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install /workspace/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

RUN git clone -b wsl --recursive git@github.com:woforce/RewardHackingMRE.git rllm
RUN cd rllm && bash setup_env.sh && cd /workspace

RUN rm -rf /workspace/rllm

CMD ["/bin/bash"]

# Docker Usage
# docker build -t rllm .
# docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
# docker start rllm-container
# docker exec -it rllm-container bash
