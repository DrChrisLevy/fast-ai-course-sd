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
sd = StableDiffusion()
```

```{code-cell} ipython3
scheduler = sd.scheduler
batch_size = 1
unet = sd.unet
height=512
width=512
generator = torch.manual_seed(32) 
tokenizer = sd.tokenizer
text_encoder = sd.text_encoder
```

```{code-cell} ipython3
!curl --output macaw.jpg 'https://images.pexels.com/photos/1996338/pexels-photo-1996338.jpeg?cs=srgb&dl=pexels-helena-lopes-1996338.jpg&fm=jpg'
```

```{code-cell} ipython3
# Load the image with PIL
input_image = Image.open('macaw.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
latents = sd.pil_to_latent(input_image)

sd.scheduler.set_timesteps(30)
noise = torch.randn_like(latents) # Random noise
sampling_step = 27
generator = torch.manual_seed(32) 
latents = sd.scheduler.add_noise(latents, noise, timesteps=torch.tensor([sd.scheduler.timesteps[sampling_step]]))
latents = latents.to(torch_device).float()
latents.shape
```

```{code-cell} ipython3
sd.latents_to_pil(latents.float())[0] # Display
```

```{code-cell} ipython3
sigma = scheduler.sigmas[sampling_step]
t = sd.scheduler.timesteps[sampling_step]
sampling_step, t, sigma,
```

```{code-cell} ipython3
guidance_scale=7
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# Prep text (same as before)
prompt = ["a zebra"]
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
sd.latents_to_pil(latents)[0]
```

```{code-cell} ipython3
noise_pred1 = noise_pred
```

```{code-cell} ipython3
noise_pred2 = noise_pred
```

```{code-cell} ipython3
contrast = noise_pred1-noise_pred2
cls = StableDiffusion
```

```{code-cell} ipython3
sd.latents_to_pil(noise_pred1-noise_pred2)[0]
```

```{code-cell} ipython3
# Let's visualize the four channels of this latent representation:
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    axs[c].imshow(contrast[0][c].cpu(), cmap='Greys')
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
