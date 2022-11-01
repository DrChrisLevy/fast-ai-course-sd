## Showing some Demos From the Week
- the example of student looking at classifier free guidance formula and how we usually use 7.5 and not something between 0 and 1. The vectors are diff in size. Can we rescale between 0 and 1. [link](https://forums.fast.ai/t/lesson-10-official-topic/101171/101?u=drchrislevy)
	- $u + g(t-u)$
	- rescale using norm of vectors

## DiffEdit Paper
-  [DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427)
- Jeremy uses some chrome extension called zotero. Or maybe its software.
	- can annotate, tag, put in folders, reading list etc. He likes it better than mendeley
- talking about going through the paper
- talking about inpainting in the intro of paper. 
- Jeremy talking about about greek letters (thankfully I know most of it) 
- and also [mathpix](https://mathpix.com/ocr?gclid=Cj0KCQjwkt6aBhDKARIsAAyeLJ3kWkmJqMGcnHZel6_QZdOO-xljQ29te47u1E1EAjevreE7Xtg7UdkaAjr-EALw_wcB) for getting the latex version of the symbols (but I guess free version sucks). other open source tools like detexify
- Downloading the latex version of the paper!
- Basically went through the example of finding out what the norm meant, expected value,
	- [expected value](https://en.wikipedia.org/wiki/Expected_value) is important concept 
- DDPM and DDIM foundational papers on which diffusion are based
- talking about the three steps 

## Took us to the 1 hour mark (took a break)

## Back to the Basics matmul notebook
- matrix multiplication with triple for loop
- now using `numbda` , [link](https://numba.pydata.org/)
	- see the notebook to speed up the inner dot product. Compiles to machine code.
- [TryAPL](https://tryapl.org/)
- Frobenius norm: $$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$
- now talking about broadcasting 
- Oh cool, `t.storage()` shows you which numbers actually stored
	- idea of strides. Tricks for broadcasting. More efficient.
	- `c[None, :]` is unsqueeze adding the unit axis i.e. `c.unsqueeze(0)`
	- `c.unsqueeze(1), c[:, None]`
- `c[...,None]`
	- unit axis at the end.
	- `...` means all preceding dimensions
- Broadcasting Rules. Study them for quiz.
- got down to right before Einstein summation

## TODO
- sign up for Jarvis labs ad get $100 credit
    - https://forums.fast.ai/t/jarvislabs-gpu-100-credit/101594
- go through the notebook 01_matmul and understand those broadcast rules and define mat mul using them. Look carefully at those broadcasting rules.
- stretch: can you manipulate and generate the mask. Part 1 of the diff edit paper.
    - read the [DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427) paper 
    - write a notebook code first approach to this paper
    - this is the diffusion homework for the week!
    - create a function to add noise to an image. See the notebook 9b with the parrot example. Add to the class. 
        - start with image of a horse like in the paper and the two queries
            - q1 = horse
            - q2 = zebra
        - add noise to image
            - predict noise needed to removed with text q1 using unet
            - predict noise needed to removed with text q1 using unet
                - also try unconditional denosing to (empty prompt?)
            - let's understand this step. Maybe make some plots and try manually removing the noise from the image and displaying it.  Maybe this is a first step to do before continuing.
            - once we have the two predicted noises p1 and p2 then compute the difference and normalize it between 0 and 1. Then binarize using a threshold 0.5 to convert to 0s and 1s. That is the mask. Visualize it!
            - From paper:  In our algorithm, we use a Gaussian noise with strength 50% (see analysis in Appendix A.1), remove extreme values in noise predictions and stabilize the effect by averaging spatial differences over a set of n input noises, with n= 10 in our default configuration. The result is then rescaled to the range [0, 1], and binarized with a threshold, which we set to 0.5 by default.
    - if we need another [scheduler](https://forums.fast.ai/t/lesson-11-official-topic/101508/41?u=drchrislevy) 
        - If you want to drop in DDIMScheduler with the stable diffusion stuff you need the right settings. I think the magic code is `scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)`
        - https://forums.fast.ai/t/lesson-11-official-topic/101508/43?u=drchrislevy
    - checkout the HF model inpainting that basically does this
      - https://forums.fast.ai/t/lesson-11-official-topic/101508/63?u=drchrislevy
- get [zotero](https://www.zotero.org/)
- start a blog!
