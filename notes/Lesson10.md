## Some Papers
Progressive Distillation for Fast Sampling of Diffusion Models
- read this paper 
- Jeremy doing a walk through of this paper to start the lesson
- teacher model and student model

On distillation of Guided Diffusion Models
- explains at a high level
- read it

Imagic: Text-Based Real Image Editing with Diffusion Models
- explains at a high level
- use the Google Model but could probably use Stable Diffusion etc.

## Back to Notebooks

- Talking about the stable_diffusion notebook again. Stuff I know from last week.
	- `.half()` converts to float16
- `shfit m` for merging cells in notebook together. NICE! Learned a new trick.
- Jeremy talking about experiment with code line by line. Understanding it. Seeing what it does.
	- we should experiment with it
	- **TODO**: try picking one of the extra tricks (negative prompt, image2image,) and do it with the 3 base models but not the Image2Image Class. 
- Jono says in chat: if the empty prompt is replaced by a non-empty one you get 'negative prompts' - a nice way to get some extra control over what you DONT want in the image.
- nice get the Navigate Table Of Contents Jupyter Extension

## Back from 10 Min Break
BUCKLE UP! Going to need Patience!
- build our own little mini-framework called `MiniAI`
- Build stuff from scratch using core Python libraries.
	- Once we implement something 
- `shift tab` nice trick to show docs for func

## Going through notebook 01_matmul

- Checkout the great itertools library
- Some history on APL
	- optional
	- where all the recent array tensor stuff comes from
- Building our own Random Number Generator
	- pseduo-random number generator
	- Based on the Wichmann Hill algorithm used before Python 2.3.
- Be  very careful and aware of the random number generator and seed.
	- initializing the random seed in each new process/fork


## TODO

- try picking one of the extra tricks (negative prompt, image2image,) and do it with the 3 base models but not the Image2Image Class. 
- PAPERS
	- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/pdf/2202.00512.pdf)
	- [On distillation of Guided Diffusion Models](https://arxiv.org/pdf/2210.03142.pdf)
	- [Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/pdf/2210.09276.pdf)
- Try implementing negative prompts, image2image, i.e. one of the tricks without using the HF PipeLIne class to do it but just using the building  blocks of the 3 main models (Unet, CLIP, VAE) and the schedulers and the diffusion loop etc. 
- Go over itertools sttuff i.e. `list(iter(lambda: list(islice(it, 28)), []))`
- go over the first matrix mult notebook that Jeremy started today.