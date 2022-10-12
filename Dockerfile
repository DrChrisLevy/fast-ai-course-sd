FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
        git \
        wget \
        htop \
        screen \
        vim

RUN pip install -r requirements.txt

# download notebooks
RUN wget https://raw.githubusercontent.com/fastai/diffusion-nbs/master/stable_diffusion.ipynb
RUN wget https://raw.githubusercontent.com/fastai/diffusion-nbs/master/Stable%20Diffusion%20Deep%20Dive.ipynb