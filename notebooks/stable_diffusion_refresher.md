---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
os.chdir('/workspace')
```

Past few weeks in the course have been working through the basics from scratch.
It's been great but I found myself forgetting what I learned about Stable Diffusion (SD).
So this notebook is simply to return to SD and play around with it a bit and remind myself
of the main parts.

```python
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from stable_diffusion import StableDiffusion
sd = StableDiffusion()
```

```python
img = sd.text_to_img(prompt = ['portrait art of 8k ultra realistic retro futuristic Gandalf, lens flare, atmosphere, glow, detailed,intricate,blade runner, cybernetic, full of colour, cinematic lighting, trending on artstation, 4k, hyperrealistic, focused, extreme details,unreal engine 5, cinematic, masterpiece, art by ayami kojima, giger ']*2,
              num_inference_steps=50, guidance_scale=7, seed=104853234)[0]
```

```python
img
```

# VAE

```python
img_tensor = to_tensor(img) # converts to image with pixel values between 0 and 1
img_tensor
```

```python
img_tensor.shape
```

We can convert our high dimensional images into a lower dimensional latent space
using the Variational Auto Encoder (VAE). Here we put the PIL image through the encoder
of the VAE (not the decoder). This his how we get the latent representation:

```python
latents = sd.pil_to_latent(img)
```

```python
latents.shape
```

```python
sd.vis_latents(latents)
```

We can decode the latents by putting it through the VAE decoder.

```python
decoded_img = sd.latents_to_pil(latents)[0]
decoded_img
```

Now, `decoded_img` will not be the exact same as the original `img` but
they will be close! Check out the abs difference:

```python
to_pil_image(torch.abs(to_tensor(img) - to_tensor(decoded_img)))
```

Because the VAE can compact the original dimension (3 X 512 X 512) into (4 X 64 X 64) it is a 48 times reduction
in pixel values. That's why we call Stable Diffusion a latent diffusion model. 


# Adding Noise to Latents

When training SD:

- start with regular images with no noise
- convert to latent space
- add noise using a noise schedule over steps
- one example of a scheduler is `LMSDiscreteScheduler`
- add noise over steps to get to an image that is all random noise
    - I guess this is the ground truth
- starting with pure noise and then stepping with the scheduler and denoising down to an image
- Unet is trained to predict the noise that needs to be removed


```python
sd.scheduler
```

```python
num_inference_steps = 50
```

```python
sd.scheduler.set_timesteps(num_inference_steps)
```

```python
# See these in terms of the original 1000 steps used for training:
print(sd.scheduler.timesteps)
```

There is more noise at the earlier steps.

```python
# Look at the equivalent noise levels:
print(sd.scheduler.sigmas)
```

This 'sigma' is the amount of noise added to the latent representation. Let's visualize what this looks like by adding a bit of noise to our encoded image and then decoding this noised version:

```python
# Here the sampling step is at the end so it is adding very little noise
sd.latents_to_pil(sd.add_noise_to_latents(latents, num_inference_steps=50, sampling_step=49, seed=42)[0][None,:])[0]
```

The noise is added in the latent space. **NOT** the image space. 
We are just putting the noised latent through the VAE decoder
just to visualize the noise in the image space.

```python
sd.latents_to_pil(sd.add_noise_to_latents(latents, num_inference_steps=50, sampling_step=40, seed=42)[0][None,:])[0]
```

```python
sd.latents_to_pil(sd.add_noise_to_latents(latents, num_inference_steps=50, sampling_step=20, seed=42)[0][None,:])[0]
```

```python
sd.latents_to_pil(sd.add_noise_to_latents(latents, num_inference_steps=50, sampling_step=0, seed=42)[0][None,:])[0]
```

Once the UNET is trained, we can use it to denoise the noisy latent to
create an image. The trained UNET predicts the amount of noise to remove. 
Then the entire process of removing noise takes place over multiple steps
and uses classifier free guidance with the CLIP text embeddings. 

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
