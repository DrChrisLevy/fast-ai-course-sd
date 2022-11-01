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
