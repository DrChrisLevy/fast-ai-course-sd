---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import os
os.chdir('/workspace')
```

```{code-cell} ipython3
from stable_diffusion import *
import numpy as np
import matplotlib.pyplot as plt
sd = StableDiffusion()
```

```{code-cell} ipython3
scheduler = sd.scheduler
batch_size = 1
unet = sd.unet
height=512
width=512
tokenizer = sd.tokenizer
text_encoder = sd.text_encoder
```

```{code-cell} ipython3
!curl --output horse.jpg 'https://forums.fast.ai/uploads/default/original/3X/7/0/709733c3ab089787b22ccc166b918738dbebfd1e.png'
```

```{code-cell} ipython3
# Load the image with PIL
input_image = Image.open('horse.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
def one_step(prompt = ["a zebra"], seed=42, sampling_step = 46, guidance_scale=7.5):
    latents = sd.pil_to_latent(input_image)
    scheduler.set_timesteps(50)
    noise = torch.randn_like(latents) # Random noise
    generator = torch.manual_seed(seed) # this is important !!
    latents = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[sampling_step]]))
    latents = latents.to(torch_device).float()
    sd.latents_to_pil(latents.float())[0] # Display
    
    sigma = scheduler.sigmas[sampling_step]
    t = scheduler.timesteps[sampling_step]
#     print(sampling_step, t, sigma)
    
    # Prep text
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return noise_pred
    
```

```{code-cell} ipython3
noises1 = []
for i in range(40):
    noise = one_step(prompt = ["a zebra"], guidance_scale=10, seed=i*100, sampling_step=20)
    noises1.append(noise[0])
noises1 = torch.stack(noises1)
noise_pred1 = noises1.mean(axis=0)[None,:]
sd.latents_to_pil(noise_pred1)[0]
```

```{code-cell} ipython3
noises2 = []
for i in range(40):
    noise = one_step(prompt = ["a horse"], guidance_scale=10, seed=i*100, sampling_step=20)
    noises2.append(noise[0])
noises2 = torch.stack(noises2)
noise_pred2 = noises2.mean(axis=0)[None,:]
sd.latents_to_pil(noise_pred2)[0]
```

```{code-cell} ipython3
contrast1 = noise_pred1-noise_pred2
sd.latents_to_pil(contrast1)[0]
```

```{code-cell} ipython3
contrast2 = noise_pred2-noise_pred1
sd.latents_to_pil(contrast2)[0]
```

```{code-cell} ipython3

```

```{code-cell} ipython3
img1 = sd.latents_to_pil(contrast1)[0]
img1
```

```{code-cell} ipython3
img2 = sd.latents_to_pil(contrast2)[0]
img2
```

```{code-cell} ipython3

```

```{code-cell} ipython3
mask = np.array(img1)
plt.imshow(mask)
```

```{code-cell} ipython3
mask.shape
```

```{code-cell} ipython3
mask = (mask - mask.min()) / (mask.max()-mask.min())
mask = np.mean(mask,axis=2)
```

```{code-cell} ipython3
mask = (mask > 0.5).astype('uint8')
```

```{code-cell} ipython3
plt.imshow(mask, cmap='Greys',  interpolation='nearest')
```
