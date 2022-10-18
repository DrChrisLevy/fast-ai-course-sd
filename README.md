# fast-ai-course-sd

## Clone Notebooks

```
git clone https://github.com/fastai/diffusion-nbs.git
```

## Build Docker

```
docker build . -t fast-ai
```

## Run Docker
```
docker run -it -p 8888:8888 --gpus all -v $PWD:/workspace fast-ai /bin/bash
```

```
huggingface-cli login
```

```
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```