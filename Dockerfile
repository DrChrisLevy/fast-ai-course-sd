FROM pytorch/pytorch
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
        git \
        wget \
        htop \
        screen \
        vim \
        curl

COPY requirements.txt /workspace
RUN pip install -r requirements.txt

COPY . /workspace