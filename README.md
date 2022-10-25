# fast-ai-course-sd

## ssh to server (optional)
Modify for your pem key and IP:

```
ssh -i "lambda-labs-key.pem" ubuntu@<ip> -L 8888:localhost:8888
```


## clone repo
```
git clone https://github.com/DrChrisLevy/fast-ai-course-sd.git
cd fast-ai-course-sd
```

## Configure Git
```
git config --global user.email christopherdavidlevy@gmail.com
git config --global user.name DrChrisLevy
git config credential.helper store
```

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