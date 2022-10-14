10 hours minimum per video/week.

Two new papers just yesterday for getting the steps down for stable diffusion from 50 to 4
	- On Distillation of Guided Diffusion  Model
	- Poisson Flow Generative Model

https://www.strmr.com/
- cool app using stable diffusion for (I think) img2img.

Compute options
- Colab, Paperspace Gradient, Lambda Labs, Jarvis Labs

Lots of links in the lesson 9 forum.
- course repo, nb repo , other videos etc. Just check it all out.
- Check out stuff from Jonathan Whitaker etc.

Going through the notebook: `diffusion-nbs/stable_diffusion.ipynb`
- guidance scale , taking the average of two images, guidance scale like a weight
- negative prompt, "blue" to remove the blue. Thats cool.
- Image2Image pipeline.
	- starts with a noisy version of the input image as a guiding point
	- strength - to what degree do you want it to link to input
	- really cool putting the output of the model back into the model as input then tweaking with a different prompt.

Back from the break

Drawing on the whiteboard
- api/function that takes input image and says probability its an image 
- all parts of stable diffusion high level whiteboard
- go over and watch again maybe..

## TODO Before Next Lesson
- set up ENV we can use for git clone on a GPU machine and Docker FIle.
- watch https://www.youtube.com/watch?v=844LY0vYQhc and work along with the [notebook](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb)
- checkout the [fast ai diffusion-nbs](https://github.com/fastai/diffusion-nbs)
- read about [diffusion models](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)  in general
- learn about this scheduler stuff!
- this guys [channel](https://www.youtube.com/watch?v=ZXuK6IRJlnk) is great
- train your own dreambooth 
- train your own textual inversion [like this script]([textual_inversion_training.py](http://127.0.0.1:8888/edit/diffusion-nbs/textual_inversion_training.py))


## Notes from diffusion-nbs/stable_diffusion.ipynb
- starting with random noise and taking steps towards a clear nice image. These models are known as diffusion models.
- Using the hugging face diffusions library there is  a safety checker and this is why we sometimes get black images. We can turn it off with 
	- `pipe.safety_checker = lambda images, **kwargs: (images, False)`
- you can adjust the number of steps. Very little steps will look noisy. More steps will look less noisy at the cost of more compute.
- classifier free guidance 
	- `guidance_scale`
	- [blog post](https://benanne.github.io/2022/05/26/guidance.html)
	- 7.5 is the default. The stronger the more closer the result to the prompt
- can do negative prompts too
- textual inversion
	- teach a new word to the text model and freeze the weights of all the models except text encoder and train with a few representative images
	- it creates a new token. Think back to the example of water color style token and then could do prompts in the style of water color painting
- Dreambooth
	- similar to textual inversion but we dont create a new token
	- instead we select an existing rare token and fine tune the model to bring that token close to the images we supply. Regular fine tuning in which all modules are unfrozen
	- the sks token for example
	- there is a cool [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb#scrollTo=rscg285SBh4M) where it shows how you can train the model to teach the token to be the several images you upload. We should try that out. 
- original diffuser models (non latent ones) operated on entire pixel space so way more compute needed for training and inference. But the latent diffusion models have the auto encoder (VAE) to compress the image into the latent space.
