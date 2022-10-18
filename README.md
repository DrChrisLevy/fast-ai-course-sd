# fast-ai-course-sd

## Clone External Notebooks
Grab any other notebooks or code you want to clone locally and do it here:

```
git clone https://github.com/fastai/diffusion-nbs.git
git clone https://github.com/fastai/course22p2.git
```

## Build Docker Image
```
docker build . -t fast-ai
```

## Run Docker Container

**CPU**

```
 docker run -it -p 8888:8888 -v $PWD:/workspace fast-ai /bin/bash
```

**GPU**

```
docker run -it -p 8888:8888 --gpus all -v $PWD:/workspace fast-ai /bin/bash
```

Now in the docker container:

```
huggingface-cli login
```

```
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```