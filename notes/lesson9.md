10 hours minimum per video/week.

Two new papers just yesterday for getting the steps down for stable diffusion from 50 to 4
	- On Distillation of Guided Diffusion  Model
	- Poisson Flow Generative Model

https://www.strmr.com/
- cool app using stable diffusion for (I think) img2img.

Compute options
- Colab, Paperspace Gradient, Lambda Labds, Jarvis Labs

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
- all parts of stavle diffusion high level whiteboard
- go over and watch again maybe..

# TODO Before Next Lesson
- set up ENV we can use for git clone on a GPU machine and Docker FIle.
- watch https://www.youtube.com/watch?v=844LY0vYQhc and work along with the [notebook](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb)
- checkout the [fast ai diffusion-nbs](https://github.com/fastai/diffusion-nbs)
- learn about this scheduler stuff!
