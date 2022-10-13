FROM pytorch/pytorch
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
        git \
        wget \
        htop \
        screen \
        vim \
        curl

COPY . /workspace

RUN pip install -r requirements.txt

# download notebooks
RUN git clone https://github.com/fastai/diffusion-nbs.git