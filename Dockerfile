# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

#RUN apt-get update && apt-get install -y git-all

RUN mkdir /workspace/ssdgm

COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt \
    && pip install https://github.com/PyTorchLightning/metrics/archive/master.zip \
    && rm ./requirements.txt

WORKDIR /workspace/ssdgm

