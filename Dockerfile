FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
        git \
        htop \
        screen \
        vim

RUN pip install -r requirements.txt