# Lesson 9

- [lesson 9](https://forums.fast.ai/t/lesson-9-official-topic/100562)
- [watch the math of diffusion again](https://forums.fast.ai/t/math-of-stable-diffusion/101077)
- Two new papers just yesterday for getting the steps down for stable diffusion from 50 to 4
    - On Distillation of Guided Diffusion  Model
    - Poisson Flow Generative Model
- https://www.strmr.com/
- Colab, Paperspace Gradient, Lambda Labs, Jarvis Labs
- Check out stuff from [Jonathan Whitaker](https://twitter.com/johnowhitaker?lang=en)
- disable safety checker: `pipe.safety_checker = lambda images, **kwargs: (images, False)`
- post some work [here](https://forums.fast.ai/t/share-your-work-here-part-2-2022/101151/17)
- [why do we need the unconditoned embedding](https://forums.fast.ai/t/why-do-we-need-the-unconditioned-embedding/101134/11)
    - [classifier free guidance](https://benanne.github.io/2022/05/26/guidance.html)
- stable diffusion [resources](https://forums.fast.ai/t/stable-diffusion-resources-and-discussion/100268/42)
- Unets and VAEs [resources](https://forums.fast.ai/t/variational-autoencoders-unets-resources-and-discussion/100269/8)

## TODO

- get dreambooth and textual inversion training working on a GPU with the HF notebooks
- find some way of writing about what you learn and sharing it. nb-dev? Quarto?
- maybe read [GLIDE](https://arxiv.org/pdf/2112.10741.pdf) paper
  - [Diffusion models explained. How does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8)
  - [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://www.youtube.com/watch?v=gwI6g1pBD84)
  - [OpenAI GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://www.youtube.com/watch?v=lvv4N2nf-HU)
- there are some useful functions in those notebooks. Maybe start creating some utils.