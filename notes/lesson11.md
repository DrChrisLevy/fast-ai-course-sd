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
- make sure these notebooks work locally on CPU and not on GPU from LambdaLabs
	- well I guess the cuda stuff wont work
- go through the notebook 01_matmul and understand those broadcast rules and define mat mul using them. Look carefully at those broadcasting rules.
- stretch: can you manipulate and generate the mask. Part 1 of the diff edit paper.
	- read the [DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427) paper 
	- write a notebook code first approach to this paper
	- this is the diffusion homework for the week!
- get [zotero](https://www.zotero.org/)
- clean up all todo lists into a master one
- start a blog!
