# 9

- https://www.strmr.com/
- Check out stuff from [Jonathan Whitaker](https://twitter.com/johnowhitaker?lang=en)
- [why do we need the unconditoned embedding](https://forums.fast.ai/t/why-do-we-need-the-unconditioned-embedding/101134/11)
    - [classifier free guidance](https://benanne.github.io/2022/05/26/guidance.html)
- stable diffusion [resources](https://forums.fast.ai/t/stable-diffusion-resources-and-discussion/100268/42)
- Unets and VAEs [resources](https://forums.fast.ai/t/variational-autoencoders-unets-resources-and-discussion/100269/8)
- [watch the math of diffusion again](https://forums.fast.ai/t/math-of-stable-diffusion/101077)
- get dreambooth and textual inversion training working on a GPU with the HF notebooks
- find some way of writing about what you learn and sharing it. nb-dev? Quarto?
- maybe read [GLIDE](https://arxiv.org/pdf/2112.10741.pdf) paper
  - [Diffusion models explained. How does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8)
  - [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://www.youtube.com/watch?v=gwI6g1pBD84)
  - [OpenAI GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://www.youtube.com/watch?v=lvv4N2nf-HU)

# 10

- try picking one of the extra tricks (negative prompt, image2image,) and do it with the 3 base models but not the Image2Image Class. 
- PAPERS
    - [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/pdf/2202.00512.pdf)
    - [On distillation of Guided Diffusion Models](https://arxiv.org/pdf/2210.03142.pdf)
    - [Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/pdf/2210.09276.pdf)
- Try implementing negative prompts, image2image, i.e. one of the tricks without using the HF PipeLIne class to do it but just using the building  blocks of the 3 main models (Unet, CLIP, VAE) and the schedulers and the diffusion loop etc. 
- go over the first matrix mult notebook 

# 11

- [DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427)
- get [zotero](https://www.zotero.org/)
- [mathpix](https://mathpix.com/ocr?gclid=Cj0KCQjwkt6aBhDKARIsAAyeLJ3kWkmJqMGcnHZel6_QZdOO-xljQ29te47u1E1EAjevreE7Xtg7UdkaAjr-EALw_wcB) for getting the latex version of the symbols (but I guess free version sucks). other open source tools like detexify
- Downloading the latex version of the paper!
- [expected value](https://en.wikipedia.org/wiki/Expected_value) is important concept 
- DDPM and DDIM foundational papers on which diffusion are based
- go through the notebook 01_matmul and understand those broadcast rules and define mat mul using them. Look carefully at those broadcasting rules.
- stretch: can you manipulate and generate the mask. Part 1 of the diff edit paper.
    - read the [DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427) paper
    - if we need another [scheduler](https://forums.fast.ai/t/lesson-11-official-topic/101508/41?u=drchrislevy) 
        - If you want to drop in DDIMScheduler with the stable diffusion stuff you need the right settings. I think the magic code is `scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)`
        - https://forums.fast.ai/t/lesson-11-official-topic/101508/43?u=drchrislevy
    - checkout the HF model inpainting that basically does this
      - https://forums.fast.ai/t/lesson-11-official-topic/101508/63?u=drchrislevy
- start a blog!

# 12 
- [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator) 
	- inverse problems 
	- not possible to invert the clip image encoder 
	- thats why the texts it spits back are fun/interesting but not actually the "correct prompt"
	- uses [BLIP](https://arxiv.org/abs/2201.12086) language model

## Matrix Mult Notebook `01_matmul`
- Einstein Summation
	- learn a little about that
	- faster than the broadcast trick
- but of course we can run it directly in torch with `matmul` or `@`
- That was CPU
- Now onto using CUDA
- a `mat_mul` that just takes a coordinate and computes just that one value of the matrix and leaves else as 0. Fills in one piece of the grid.
	- can do in parallel b/c all independent 
	- uses GPU kernel 
	- `from numba import cuda`, `@cuda.jit`
	- `cuda.to_device()` to put the devices on the GPU.
	- also a concept of `blocks`, `TPB`, threads per block
- was so much faster on the GPU. A 5 Million X change in this

## Mean Shift Notebook
- we need to be good at all the matrix multiplications, broadcasting, mechanical operations , tensor operations etc. Practice Practice Practice! 
	- That is the purpose of the mean shift clustering notebook. To practice the things we learned in the mat mul notebook
	- Also see the HW suggestions on implementing some stuff on just GPU
- `functools.partial` , uses a lot. In this example was used for the plotting. 
- `X = data.clone()`, pytorch thing
- go over those broadcast rules!
- [**norm**](https://en.wikipedia.org/wiki/Norm_(mathematics))
	- comes up all he time when dealing with distances, loss functions etc.
	- l2 norm is euclidean 
- matplotlib FuncAnimation 
- broadcasting on GPU, do things in batches 
	- does the mean shift over again on GPU with batches
	- those broadcasting rules are really important!

## Calculus
- you use to teach that but would be good to refresh on the Matrix Calc stuff.


## TODO 
- rewrite l2 norm euclidean distance with ein sum notation
- implement k-means clustering, dbscan, locality sensitive hashing, or some other clustering, fast nearest neighbors, or similar algorithm of your choice, on the GPU. Check if your version is faster than a pure python or CPU version.
- Invent a new meanshift algorithm which picks only the closest points, to avoid quadratic time.
- make an animation
- go over those notebooks
- [essence of Calculus by 3Blue1Brown](https://www.youtube.com/watch?v=WUvTyaaNkzM)
- [chain rule intuition](https://webspace.ship.edu/msrenault/geogebracalculus/derivative_intuitive_chain_rule.html)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)
- I havent learned the Einsum notation yet. Go back and study that part.
- Go back and study the Cuda section of the 01 mat mul notebook
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) by Andrej Karpathy
- [Einsum](https://ajcr.net/Basic-guide-to-einsum/)
  - another [resource](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/) teaching Einsum notation
- would be really fun to implement [product quantization](https://www.pinecone.io/learn/product-quantization/) with torch and GPU and then plug in with my Kmeans GPU and then try to improve mean shift with this approx NN search etc.

# 13
## backprop Notebook
- starting off with 03_backprop notebook
- stack of two layers with MNIST and simple MSE loss
- going over derivatives and gradients background knowledge 
- cool little trick of using `import pdb; pdb.set_trace()` interactive debugger within the notebook. Can also achieve same thing with `breakpoint()`. `c` to continue and `n` for next line. Actually `breakpoint()` does not work with Jupyter notebook. 

### Refactoring the Above BackProp with Layers as Classes
- see the notebook. `__call__` as the forward pass and `backward` as the method for backward pass and using `self.blah` to store the layers and intermediate steps etc.
- we did it this way so then we could subclass `nn.Module`
	- definitely spend some time practicing that 

## MiniBatch Training NoteBook
- talking about how we could use a different LOSS function. MSE not the best approach for that classification. 
	- softmax function,  cross entropy loss, log softmax
- Log Sum Exp trick . That's cool. Avoid super high numbers and floating point issues.
- `nll_loss` negative log likelihood loss, log_softmax, and cross_entropy that puts them together. Need to study this and go over it slower. All come from `F` module 


## TODO
- at some point read [# The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)
- do the 03 backprop notebook on your own
    - feel free to do some parts slightly differently.
    - the main objective is to do some forward and backward passes and derivatives from scratch without using autograd. 
    - Write the math out on paper. I honestly find it easier to code up after seeing it on paper first. Put in image of that in the notebook. 
    - PRACTICE PRACTICE the subclassing `nn.module` and looking at some simple examples on torch docs
    - review/practice some basic train loops in torch
- review some basic softmax and cross entropy stuff, nll_loss, log_softmax, cross_entropy
    - implement yourself and compare with torch 
- good docs on [nn.module](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
- From last week go and try the product quantization. Does it male Mean shift quicker?
- Also see TODOs from last week you didn't do yet!
- Someone recommended this [VIDEO](https://www.youtube.com/watch?v=dB-u77Y5a6A&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r&t=2312s) for backprop and 
    it was amazing. Especially for higher dimensional Jacobian's and some simple heuristics and rules for getting shapes to match up etc.
- This video was good
- someone said check out [Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
- [The spelled-out intro to neural networks and backpropagation: building micrograd
](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5836s)
  - Actually he has been doing a [series](https://www.youtube.com/c/AndrejKarpathy/videos) 