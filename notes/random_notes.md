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

# 14

- started with a review of last time about back propagation and those partial derivatives involved in the linear layer
- Going through `04_minibatch_training.ipynb` notebook
- talking about `nn.module` and implementing a simple class of what it does before we use it in torch
- Now talking about `nn.ModuleList` and registering Modules 
- `nn.Sequential`, similar to `tf.keras` sequential 
- Optimizers
	- implementing our own first and then just calling `torch.optim.SGD`
- `Dataset` class 
	- our own simple dataset class first from scratch
- DataLoader
	- a iterator 
	- again, always looking at how we can implement a simple class on our own first
	- random sampling, samplers, batch samplers, `collate_fn`
	- multi processing data loader 

##  Hugging Face Datasets

- `05_datasets.ipynb`
- turning HF dataset into torch data loader with some collate function
- `with_transforms`
- python `@inplace` decorator 
- `itemgetter`
- torch default collate function `default_collate`
- `collate_dict`

- now talking about building MiniAI with nb-dev and `# export` in some of the notebooks
	- just the first mention of it and a heads up that we will be using nb-dev etc.
- `fc.delegates` cool trick/tool for `kwargs`
- going through some plt stuff for showing images, axis, grids, subplots, etc.

##  06_foundations.ipynb

- some foundations we need for the future notebooks 
- CallBacks
	- passing in a function that some other function will call at particular times
- lambdas and partials 

## TODO
-  how to make torch run better on mac m1 within docker 
	- is there arm docker images?
	- Right now `uname -m` is printing `x86_64` and emulation is slow!
- learn about torch datasets and data loaders
	- `DataLoader, SequentialSampler, RandomSampler, BatchSampler` etc.
- understand the data loader stuff in `04_minibatch_training.ipynb` notebook
- go through the `05_datasets.ipynb` notebook
- go through `06_foundations.ipynb`
- HW - debug those notebooks and understand stuff under the hood
- practice some more training loops with optimizers
- Jonathan Whitaker video [fun with optimization](https://youtu.be/rO5nmpniYkU) 
	- [forum link](https://forums.fast.ai/t/fun-with-optimization/102045)

# 15 
## 07_convolutions.ipynb

- convolutions explained
	- [link](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)
- going through example of 3by3 edge detector kernel
- talking about [img2col](img2col convolution) convolution algorithm which turns it into a Matrix Mult under the hood
- [`F.unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html )
- [F.conv2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html)
- strides and padding
	- reduce dimensionality of input is super useful such as auto-encoder
- Note there is this ideas of grouped convolutions and `groups` argument.
	- In most applications Ive seen `groups=1`

- half way point break

## 08_autoencoder
- need to review some things :) 
- warm up classifier code on fashion mnist 
- autoencoder
	- stride 2 compressing in the first part
	- TF Keras2 book had some good stuff on these convs for upsampling in autoencoder like networks
	- nearest neighbour upsampling with stride 1 convolution --> double grids size
	- a deconvolutional layer
	- `padding=ks//2`
- see new fit and eval
- So Jeremy was trying to do an autoencoder on mnist
	- not working to well
	- good illustration basically
	- code was training slow (cpu bottle necks etc)
		- dataloader was slow
	- Not fast enough to experiment on
	- So the take away is **don't work like this**
	- Have code that you can run hundreds of experiments on quickly.
		- runs fast but can also iterate fast.
	- This motivates the 09_learner notebook
		- we need our own custom learner 
		- got to stop doing things manually

## 09_learner
- start off at 1:35 in video.
- going through the learner class
	- the first one is only good for classification
	- not flexible enough for our purposes
- metric class (base metric class)
- `@property` is nice.
- now onto new learner with callbacks :) 
	- `with_cbs`
- lots of nifty python going on here
	- like back to that 06foundations stuff
- I def need to practice the call backs and decorators etc.

## TODO
- I wrote out some notes on convolutions once, would be nice to find them
- Would be good to find a simple little problem to work on to practice
    - dataset and dataloader again to refresh memory from lesson14
    - simple conv network with some RGB images
    - simple train loop
    - all from scratch just to practice 
    - I think the classifier would be good on fashion mnist for example or any other small dataset from HF or whatever. To practice collate , ds, dl, train, eval, etc.
- review some stuff from the 08_autoencoder notebook that was assumed remembered from last week
    - go through from scratch 
- go through the 09_learner notebook
    - learn more python! callbacks, decorators, etc.
    - build something simple with some of these ideas
        - partials
        - __getattr__
        - callbacks
        - decorators 

# 16

## 09_learner
- seems like he redid this so lets start over this notebook from scratch
- almost first half of this lesson was going through all this stuff and callbacks
- go through this carefully
- learning rate finder

## 10_activations
- the higher the learn rate, often can find more generalizable set of weights 
	- and training with less batches less likely to overfit
	- stable training
- Looking Inside the Model with Hooks
- plotting activations (means and std)
    - want the mean of activations close to 0 and std close to 1. Basically want a nice spread. 
        - Not all zero
        - Classic shape of bad behaviour 
    - want mean around 0 and variance around 1 for the activations to be training properly
    - To be clear, the activations here are the output of `layer(X)` for each layer. For each
          forward pass we can append the value of `layer(X)`for each batch call and take the mean/stf
  		  and append to a list. So each layer has its own list of means/stds.
- All about looking inside the models to debug them. 
- In the first example he used a Sequential Model and hardcoded in the collection of stats manually but we dont have to do things manually. We can use pytorch **hooks**
- forward hooks and backward hooks. Can add them to models / layers.
	- a forward hook calls a function during the forward pass etc.
	- `m.register_forward_hook`
	- do they have hooks in TF?
- hooks and callbacks are the same thing but pytorch calls them hooks. Particular kind of callback.
- __del__ and self.remove() to delete hook after done to free up memory
- these callbacks and hooks are really cool but need to practice 
- context manager hooks class
	- review those context managers, __enter__, __exit__, etc. `with`
- histograms with the single columns of pixels and log values
- 

## TODO
- go over the 09 lesson notebook
  - do lots of it from scratch
	  - update: only have the time to do a lot of copy/paste and running code right now.
		  - There is lots of nice ideas to come back to here.
- 
- can we get the warmup conv classifier I was doing last lesson to work?
  - Can we get the auto encoder to work that I never started

# 17
- some changes to learner, callbacks, and hooks
	- go over those notebooks again

## 11_initialization notebook
- fashion mnist with simple conv
- the activation stats is showing those spike and things are not going well during training
- clean ipython hist utility function
	- also a utility function about cleaning tracebacks
	- these are all about clearing out cuda memory etc when training in notebooks
- why do we need 0 mean and 1 standard deviation on those activation stats?
- have to scale the weight matrices just right
- [paper about this](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), Xavier Glorot, Yoshua Bengio
	- because of this work we now have a way to initialize nn weights 
- variance is sensitive to outliers because of the square term
	- going over basic mean and std
- covariance - how much 2 things vary .Maybe go over that.
- pearson correlation coeff
- Xavier weight initialization 
- but doesnt work wirth relu activation
	- matirx mult and relu means variance and mean go down
- Kaiming came up with a modified approach, [paper](https://arxiv.org/abs/1502.01852)
	- 	- `init.kaiming_normal_(m.weight)`
- its quite nice, now the lr finder started working 
- but still not training the greatest. Still not working. We forgot something critical
	- the weight matrices are correctly normalized now with the Kaiming method 
	- but not the input!. So need to modify the input so the inputs have mean 0 and variance 1. Can do with a callback that works on each batch actually.
	- and now its training even better with this change
	- stats color dim now looking more rectangular 
	- pretty cool!
- despite normalizing the weights and the inputs, we still not have a mean of 0 and a var of 1. The problem was we are putting the data through a Relu. Relu removes all neg so its impossible for the output of Relu to have a mean of 0. Jeremy think's its incompatible . Came up with idea of taking result of Relu and subtracting something from it. Pull the whole thing down so there are some negatives. And also, while we there use a Leaky Relu.
- General Relu
	- need to make an adjustment when using kaiming_normal with Leak Relu. `a` argument
	- and training even better!
- Now a days, much more complicated activations. Not just relu and leaky relu etc.
	- Need to initialize weights correctly to keep the mean 0 and var 1. Jeremy says many people dont do this part correctly.
	- [paper: all you need is a good init](https://arxiv.org/abs/1511.06422)
		- general way of initializing neural network weights that works for any activation
			- LSUV - layer-wise Sequential Unit Variance (LSUV)
			- iterative approach. batches, for each layer, bias-mean and weight/std. See notebook/paper for details
		- used a hook. Did not do any weight initialization. Just used LSUV.
- Batch Normalization
	-  very similar in idea to LSUV
	- [2015 paper super famous](https://arxiv.org/abs/1502.03167)
	- we are normalizing the layers weights distributions  but the layers inputs **change** during training. Can fix this issue by normalizing inputs during training
	- [layer normalization paper came out year later](https://arxiv.org/abs/1607.06450) and is actually a bit simpler. Start with that.
		- start with this b/c its a simpler technique
		- see simple LayerNorm class in the notebook
		- OH interesting, `x*self.mult + self.add` where `self.mult` and `self.add` start off as 1 and 0 but they are parameters! They are learned during training! Very cool idea.
		- See the forward pass for the details.
		- added layer normalization to each layer except the first one
		- but these layer normalizations can cause some challenges.
			- first batch norm created complexities
			- trend in recent years to not rely so heavily on normalization layers
			- not a silver bullet
	- Now lets look at Batch Norm
		- got the mult and add like before. But now not just one. There is now one for every channel. more learnable params. 
		- but also another difference. See the update_stats method.
			- get the mean per filter and then use lerp. exponentially weighted moving average
			- only updated during training
			- during inference BN layers are frozen
		- remember that BN makes things tricker when doing transfer learning
	- See the nice image diagram from the [group norm paper](https://arxiv.org/pdf/1803.08494.pdf) which shows the batch norm, later norm, instance norm, group norm
- towards 90%
	- lower batch size to see more samples. Ha very subtle. lower batch size take longer but sees more of the data
	- Momentum learner, fine tune with a smaller lr for final epochs
- and that is the end of initialization, an incredibly important topic

## 12_accel_sgd
- weight decay
- l2 regularization
- momentum
- RMSprop
- Adam (RMSprop + Momentum)



## TODO
- go over those learner callbacks hooks notebooks more carefully and understand it all. I fell behind on those.
- covariance , pearson correlation coeff
	- go over in code and math, 


# 18 
## 12_accel_sgd
- learning rate scheduler
- 1 cycle training
	- [paper](https://arxiv.org/abs/1803.09820)
	- See the plot of learning rate and momentum over time
- Fixup initialization 
	- [paper](https://arxiv.org/abs/1901.09321)
- T-Fixup
	- [link](https://paperswithcode.com/method/t-fixup)
- Basically there all all these tricks for initializing models

- Back after the break
	- Notice the difference with the `learn` being passed to the callback
	- Oh cool Github Trick, put `/compare` at the end of the PR view URL. So replace `/pulls` with `/compare` to quickly see changes.
	- `@fc.patch` to add a new method to a class. For example `lr_find`

## 13_resnet
- in general, more depth and more layers gives the network more ability to learn
	- in our example made it deeper/wider and got better accuracy
- But there comes a point/time where adding more layers becomes a problem
	- See this [paper](https://arxiv.org/abs/1512.03385)
	- Even became worse on the training. The 56 layer was worse than the 20 layer. This team came up with a really important insight. 
	- They added something called shortcut connections 
	- Instead of being `out=conv2(conv1(in))` lets do `out=conv2(conv1(in)) + in`
		- This is the idea of the skip connection
	- A network that is deep but at the beginning of training it behaves like a shallower network. The `residual` is `out-in` so it's called as **ResnetBlock**
	- Only works if you can add those things `conv2(conv1(in))` and `in` together.
		- The answer is to add a `conv` on `in` and to make it as simple as possible. The simplest most possible conv is a 1by1 kernal size.
	- See code (`_conv_block`) function.
	- I think all these details are explained really well in François Chollet 
- So now stacking the resnet blocks together
	- notice the new utility functions for printing model shape and summary
	- from 91.7 to 92.2 acc
	- Simplest possible Resnet block
- Also tried the `timm` library (not much better results)
	- goes to show that thoughtful design and architecture goes a long way
	- All about using common sense. Encouraging.

## 14_augment
- data augmentation
- But before we discuss data augmentation
	- some improvements to the model
	- Made it a bit wider
	- change the kernel window size 
	- replace the awkward flatten with the more usable global average pooling
	- Improve the model summary mflops - basic idea counting number of multiplications
	- played around with some different architectures 
	- lowered the total number of params and flops and kept the same accuracy. A good take away is to always look to make the models smaller and faster (less compute).
	- train for 20 epochs and its basically starting to memorize the training data
	- Need more regularization
- With batchnorm, weight decay does not really regularize. Lets Look at Data Augmentation.
- data aug not typically done on eval set 
- batch transform call back
- random crop with padding
- Data aug happening on the GPU.
- Got to 93.8%
	- emphasize: using all the standard tricks 
- Test Time Augmentation (TTA)
- Random Erase
	- delete a little bit of each pic and replace with some noise
	- but replace with noise that has the same stats (mean and std of the image). Because we don't want to change the stats of the image.
	- then does a way (clamp) as not to change the range of the pixel values
- Random Copy
	- same idea. random copy part of image to another.
- Keep seeing the idea of testing line by line in the repl and then after making a function. I feel like I already do this a lot in ipython shell.
- ensemble - trained 2 models and took the average
- HW - try doing your own scheduler, try and beat on MNIST Fashion 5, 10, 20 epochs. Ideally with MiniAI.

# 19
- todo lesson 19 notes
- Introduced DDPM paper and code

# 20
- Jeremy starting with how to do mixed precision
- talking about [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index)
- Then onto 6A_StyleTransfer By [Jono](https://twitter.com/johnowhitaker)

## 6A_StyleTransfer
- starting with example of random noisy image and training a simple model with MSE loss
with the target being the image of the lady with the sunglasses. And we can train the
model to take the noisy image and output the lady with the sunglasses model. This is more like
a unit test example.
- Now at the 35 minute mark or so and talking about VGG feature extractor
  - reminder of [feature visualization](https://distill.pub/2017/feature-visualization/)
- cool idea here
  - take a noisy random image and a target image.
  - extract some features by feeding both noisy input and target image through feature extractor.
  You choose here which layers to use. The layers you pick has an effect on the final outcome.
  - Train the model with MSE loss and use the features extracted in the loss. So you are training
  the input image to have extracted features similar to those of the target. The idea is that
  you will not get an exact replica of the target image in the end like we did with the unit
  test example.
- Now notice that it is still using spatial features. Meaning the thing we get after training
is very spatially similar to the target image. But when we say we want a "style" we don't really
want to copy the exact spatial layout of the target image. We just want the "style"
- This is where the **Style Loss with Gram Matrix* comes into play
  - [see this older paper](https://arxiv.org/pdf/1508.06576.pdf)
  - [Gram Matrix][https://en.wikipedia.org/wiki/Gram_matrix]
  - see StyleLossToTarget so learning to match the gram matrix of the features


## 16B_NCA.ipynb
- [paper](https://distill.pub/2020/growing-ca/)
- [self organizing textures](https://distill.pub/selforg/2021/textures/)
- neural cellular automata
- gradient normalization is a cool trick

## TODO
- try the hooks feature extractor i.e. Homework: Can you do this using hooks 
- learn more about the gram matrix and how we could use it

# 21
- started with Jono looking at an example with [cifar](https://huggingface.co/datasets/cifar10) dataset
- start learning and playing with [weights and biases](https://wandb.ai/home)
- when training starts to take like 10 minutes, 1 hour, etc. then worthwhile setting up some
infra to do experiments 
- Jeremy does make a good counterpoint though about not going "crazy" and running hundreds of
experiments. He makes the point of making strong hypothesis, refactoring code, and making
changes that way is his preferred approach.

## 18_fid.ipynb
- Jeremy talking about research they have been doing
  - is there a metric for how good a generated image is?
  - FID metric in this [notebook](https://github.com/fastai/course22p2/blob/master/nbs/18_fid.ipynb)
  - Rather than directly comparing images pixel by pixel (for example, as done by the L2 norm), 
  - the FID compares the mean and standard deviation of the deepest layer in Inception v3. These layers are closer to output nodes that correspond to real-world objects such as a specific breed of dog or an airplane, and further from the shallow layers near the input image. As a result, they tend to mimic human perception of similarity in images
  - Talking about the GlobalAvgPool feature extractor and taking the mean over a batch
  - metric across a bunch of images
  - taking the pre-trained mnist model from notebook 14
    - note the difference in scaling between -1 and 1 (common in diffusion models) as the
    - expected input
  - now take the generated images from DDPM notebook model we trained 
    - note bug where the output was between 0 and 1
    - now pass the generated models from the generative DDPM model and pass them through
    the trained classifier but then extract the embeddings 512 before the linear layer
      - shows several ways of doing this
  - so we get the features form the generated DDPM images and also from a bunch of real fashion
      images
  - The Frechet Inception Distance score, or FID for short, is a metric that calculates 
    the distance between feature vectors calculated for real and generated images
    - [wiki](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)
  - taking the mean of all the embeddings is not enough!
  - we need to use the covariance matrix. So we get a 512by512 matrix for the covariance matrix
of the **real** images and another matrix for the **generated** images.
  - Looking for means to be similar and the covariance matrices to be similar. Goes into FID
  score.
  - some caveats of FID is:
    - depends on number of sample points.
    - When using inception network comparision (common in papers) resizing to 299by299 
    does not always make sense (think small images upscaled or large generated down scaled)
- Another metric that is less biased is KID (Kernel Inception Distance)
  - see [here](https://machine-learning-note.readthedocs.io/en/latest/generative_models/metrics.html)
  - It has a very high variance though. Jeremy has not found it useful.
- So we don't actually have an awesome metric. But sticking with FID for now and wrapped
it up in a class.

- now Jeremy talking about the BUG again of having things between 0 and 1 but everyone else
scales the inputs between -1 and 1. And he fixed the bug and things got worse. So he spent
3 days going through everything from scratch. All the notebook cells etc. Then came up the
interesting question: Why does everyone scale the inputs between -1 and 1?
- we know having centered data between -1 and 1 is good.
- instead of going from 0 to 1 then try -0.5 to 0.5
  - FID was dramatically better
  - built up a deep intuition of how things were working
- so playing with the input scaling
- started playing with different schedulers for DDPM, betamax, alpha_bar, etc.
- started looking at DDIM
- DDIM paper 
- Code for DDIM
- using the scheduler from diffusers library and swapping in DDPM or DDIM. 
- DDIM is faster and actually simpler to implement. 
- Jeremy shows some different examples and simulations for different schedules and parameters and the resulting FID score and images.  

## TODO
- learn that DDPM and DDIM stuff.
- Get comfortable with all the code.

# 22

## 22_cosine.ipynb
- talking about DDIM again. diffusion schedulers. The noisfy function is changing a little.
Talking about using a cosine schedule. Changing the process to a more continuous time value
between 0 and 1. As opposed to discrete timesteps. The model, callbacks, fitting process are the 
same still. Sampling now uses linspace. Getting a FID of 3 with 100 steps and seems like a good
sampling approach.
- Dont have to be bound to 1000 time steps like original DDIM paper. Can reframe from 0 to 1
for example. Simpler notations over time is better as people understand the problem better.


## 22_noise-pred.ipynb
- Jeremy talking about some new recent research by his team
- create a model to predict T given the noised image. Can the model actually figure out
the amount of noise instead of passing a T like we usually do.
- model to predict alpha_bar_t. Predict noise amount.
- remember to always have a baseline model!
- doing a regression problem here in this research.
- Same old ResBlock Conv model. The output is now different (single number regression model).
  - Learner uses MSE as loss.
  - predicting alpha_bar_t
- Hypothesis was correct. We can predict the thing we were manually plugging in as input.
Would be simpler if we did not have to pass the T each time.
- So now try a "No Time" model. Now nosify does not return T anymore. We dont know it.
The Unet does not have T so just passes 0.
- Copy pasted the cosine notebook and ran this new thing.
- Turned out pretty Garbage with FID of 22. 
  - Dont give up!
- Try something else. 
  - use estimated alpha bar t median for batch clamped not to be too far away
  - FID 3.88
- These no T approached could pass the T approaches. He only spent couple days on this. Early days.
- Cool to highlight the research process.

## On the importance of Noise Scheduling for Diffusion Models
- The other research area. Remember the bug from -1 to 1 and switching to -0.5 to 0.5.
    How do we normalize? Etc..
- Found a recent paper that shed some light on this. 
- [On the importance of Noise Scheduling for Diffusion Models](https://arxiv.org/abs/2301.10972#:~:text=There%20are%20three%20findings%3A%20(1,scaling%20the%20input%20data%20by)
  - A really nice paper.
  - the same noise level for different size/resolution images can be very different
  - signal to noise ratio. Different schedules for noise schedules. Linear/Cosine/Sigmoid etc.
  - there is no one true best schedule 
  - schedule and how to scale/normalize image

## 23_karras
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364.pdf)
- karras one of the authors
- maybe predicting the noise is sometimes a bad idea
- predicting the noise or predicting a clear image. One can be better in diff scenarios.
- predict a lerped version of the noise and clear image based on how much noise. See equation
in notebook.
- its a more general thing here called the V objective. Stable diffusion 2.0 and other new models
are using this.
- Mean 0 and variance 1 really important for the input data and activations. Keeps coming up. 
- just going through the notebook and paper
- Euler Sampler --> FID 1.98
- sample Heun

# 23

## 24_imgnet_tiny
- working with tiny imgnet dataset, 200 categories
- create dataset which gives the path of the image file and the name of the category that its in
- read img 3 channels, divide by 255, then normalize by subtracting mean and dividing by std
- the usual stuff by going through it manually to practice (setting up Dataset and data loader)
- 64by64 px and very easy to overfit. Found data augmentation was required.
  - Padding, random crop, random flip, random erase
  - put them data augs together with a nn.sequential (could have used torch transpose)
- used basic convnet with some resnet blocks
  - took a while to get to this point
  - 60%
- going deeper
  - resblocks 
  - got up to 62%
- more augmentation
  - [TrivialAugment](https://arxiv.org/pdf/2103.10158.pdf)
  - not as well known but it should be :) 
  - actually built into pytorch
- pre-activation resnets
  - [see here for example](https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e)
  - 65%
  - 67.5% on 200 epochs

## 25_superes

- super resolution (not classification)
- trying to go from 32by32 to 64by64 
- also added rand_erase on training to the model can learn a little more about filling
missing parts
- img in and img out for this model
- going to use the UNET
- start with "dumb" denoising autoencoder
  - F.mse_loss
  - output predicted images look pretty terrible
  - sort of like earlier auto encoder
  - really challenging
  - small data, simple model
  - Maybe it would be possible with much more data and fancier model etc.
  - but no reason for us to do that here
  - note that Stable Diffusion has the Latent Diffusion modle which trained an encoder/decoder
  that does an amazing job
  - But instead of getting all complicated we can try a UNET
- UNET (1 hour mark video)
  - [original paper](https://arxiv.org/pdf/1505.04597.pdf)
  - a UNET is actually used in Stable Diffusion to Predict the Noise
    - Remember Stable Diffusion as the latent encoder, the CLIP model, and the UNET.
  - The trick is the copy/crop over the activations from the downsampling path and concatenating
  with activations from the upsampling path. Half are from upsampling and half are from downsampling.
  - See the typical diagram that looks like a U with the arrows going across.
  - can even use adding instead of concat. Sort of like a boosting method.
  - `nn.ModuleList`
  - in the forward pass need to keep track of activations in order to copy/paste over
  - See the code. It's actually pretty easy to define. Super cool idea!
- Perceptual Loss
  - what was that about again?
  - getting the extracted features (256, 8, 8)
  - using for the perceptual loss
- then a trick with using a pre-trained model and unfreezing
  - take UNET
  - self.start =  actual weights of pre-trained model and for downsampling
  - `load_state_dict` and `state_dict`
  - turn off requires_grad
  - the classic fine tune approach
  - intermediate layer activations
  - its like cheating, but okay to do that here in this case!

# 24
- 26,27,28 notebooks
- I have bad notes for this lesson because was taking on my iphone
- lots of good UNET torch code here so good to back over and understand
- Good to come back to see how Attention works in the context of images
- UNET unconditional 
- Embedding ResBlock. Time embeddings. Sinusoidal embeddings. 
- Timestep embeddings 
- [U-Net model for Denoising Diffusion Probabilistic Models (DDPM)](https://nn.labml.ai/diffusion/ddpm/unet.html)
- Swish activation 
- Non attention stable diffusion unet. Our very first one from scratch. 
- Now looking at attention.
- AttentionBlock explanation and from scratch. 
- Self attention multihead 
- All this the context of images though and not sequences from NLP
- Eionops library
- last 5 minutes is the conditional version

# 25
- final lesson
- next course going to be on NLP

## Simple Diffusion Audio
- Jono doing this part of the lesson
- audio clips of birds
- 30000 values/sec audio data
  - too big to work with and sound waves have high frequency components
  - do the usual mel spectogram trick and turn the audio 
  - I have done this before with Whisper and stuff I did at work
  - this is really cool: https://musiclab.chromeexperiments.com/Spectrogram/
  - turns huge waveform into 128 by 128 1 chanel image
  - using mel class to do this stuff
  - lossy conversion of the sound
- adding random noise to these generated images, using SimpleDiffusion Class, UNET, transformers block
  - from the last lesson. Go over this stuff again, pretty cool!
  - SimpleDiffusion only came out about a couple months ago
- Diffusion to generate new log mel spectograms diffusion, diffusion on spectograms
  - https://www.riffusion.com/
  - pretty cool stuff!

## 29 VAE Notebook
- 256,256,3 channel to 32,32,4 in the latent space.
- going back to first building a simple auto encoder (remember we did this way back)
  - this time around much better then previous attempt 
  - but cant generate good looking or new images from random noise
- now the VAE
  - `mu,lv`
  - normally distributed numbers
  - generate encode is now a little random
  - binary cross entropy loss with logits
    - train it a certain way then will just behave like the original auto encoder. 
    - Not what we want
  - so how do we get it to create a log variance that does not go to 0?
    - by using the kld loss Kullback–Leibler divergence
    - so the space around a point should map back to that point
      - any point within the range should go back
      - get a nicer latent space
    - mapping to single vector compression
  - now the Stable Diffusion VAE
  
## Notebook 30 Bedroom Dataset
- LSUN https://www.yf.io/p/lsun
  - Jeremy put smaller subset on aws s3 
- absolutely remarkable that we can recover back the original image when we decode
  - it's hard to really tell the difference
- also had a perceptual loss
- memory mapped numpy file for using disk for storing stuff in memory (.npmm) - little trick
- generating some bedroom pics using the VAE from diffusions library and then the model we 
wrote from scratch for training
- also did 31 notebook
