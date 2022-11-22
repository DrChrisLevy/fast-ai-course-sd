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
from stable_diffusion import *
sd = StableDiffusion()
```

```python
img = sd.text_to_img(prompt = ['portrait art of 8k ultra realistic retro futuristic Gandalf, lens flare, atmosphere, glow, detailed,intricate,blade runner, cybernetic, full of colour, cinematic lighting, trending on artstation, 4k, hyperrealistic, focused, extreme details,unreal engine 5, cinematic, masterpiece, art by ayami kojima, giger '],
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


Lets start with an image and a little bit of noise and use the Unet to predict the amount of noise to remove.

```python
noisy_img = sd.latents_to_pil(sd.add_noise_to_latents(latents, num_inference_steps=50, sampling_step=47, seed=42)[0][None,:])[0]
```

```python
noisy_img
```

```python
latents_from_noisy = sd.pil_to_latent(noisy_img)
```

<!-- #region -->
The Unet takes in the noisy latents and predicts the noise. We use a conditional model that also takes in the timestep `t` and our text embedding (aka encoder_hidden_states) as conditioning. Feeding all of these into the model looks like this:

`noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]`

The Unet requires the following inputs to predict the noise:

- `latent_model_input`
- `t` (which step in the scheduler process) 
- text_embeddings


Given a set of noisy latents, the model predicts the noise component. We can remove this noise from the noisy latents to see what the output image looks like (`latents_x0 = latents - sigma * noise_pred`). 
<!-- #endregion -->

```python
_, text_embeddings = sd.embed_text("")
```

```python
noise_pred = sd.unet(latents_from_noisy, 47, encoder_hidden_states=text_embeddings).sample
```

```python
sd.latents_to_pil(latents_from_noisy - sd.scheduler.sigmas[47] * noise_pred)[0]
```

<!-- #region -->
Above we did not use the classifier free guidance and the guidance scale parameter.


By default, the model doesn't often do what we ask. If we want it to follow the prompt better, we use a hack called CFG. There's a good explanation in this video (AI coffee break GLIDE).

`noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)`

Also a note on sampling and scheduler:
How exactly does the sampler go from the current noisy latents to a slightly less noisy version? Why don't we just use the model in a single step? Are there other ways to view this?

The model tries to predict the noise in an image. For low noise values, we assume it does a pretty good job. For higher noise levels, it has a hard task! So instead of producing a perfect image, the results tend to look like a blurry mess. So, samplers use the model predictions to move a small amount towards the model prediction (removing some of the noise) and then get another prediction based on this marginally-less-rubbish input, and hope that this iteratively improves the result. Different samplers do this in different ways. 
<!-- #endregion -->

So here is what the diffusion loop looks like when we start with a text prompt.
First we come up with the prompt and choose the inference steps, guidance scale and the seed:

```python
prompt = ['portrait art of 8k ultra realistic retro futuristic Gandalf, lens flare, atmosphere, glow, detailed,intricate,blade runner, cybernetic, full of colour, cinematic lighting, trending on artstation, 4k, hyperrealistic, focused, extreme details,unreal engine 5, cinematic, masterpiece, art by ayami kojima, giger']
num_inference_steps=50
guidance_scale=7.5
seed=104853234
```

Compute the CLIP embeddings for the text prompt:

```python
_, text_embeddings = sd.embed_text(prompt)
```

```python
text_embeddings.shape
```

We also compute the text embeddings for an empty string prompt.
We concatenate these together so we can do one forward pass with the unet.

```python
batch_size = text_embeddings.shape[0] # 1
_, uncond_embeddings = sd.embed_text([""] * batch_size)
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
text_embeddings.shape
```

Now we start with some noisy random latent:

```python
# start with noisy random latent
latents = torch.zeros((batch_size, 4, sd.height // 8, sd.width // 8))
sd.scheduler.set_timesteps(num_inference_steps)
latents = sd.add_noise_to_latents(latents, num_inference_steps, 0, seed)
latents.shape
```

```python
sd.latents_to_pil(latents)[0]
```

```python
# Loop
all_latents = []
all_noise_preds = []
with autocast("cuda"):
    for i, t in tqdm(enumerate(sd.scheduler.timesteps)):
        # one X latent for the text prompt and one for the "" prompt
        latent_model_input = torch.cat([latents] * 2) 
        latent_model_input = sd.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = sd.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = sd.scheduler.step(noise_pred, t, latents).prev_sample
        
        all_latents.append(sd.latents_to_pil(latents)[0])
        all_noise_preds.append(sd.latents_to_pil(noise_pred)[0])
```

```python
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(50, 50))
columns = 5
rows = 10
for i in range(1,51):
    imgg = all_latents[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(imgg)
plt.show()
```

```python
fig = plt.figure(figsize=(50, 50))
columns = 5
rows = 10
for i in range(1,51):
    imgg = all_noise_preds[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(imgg)
plt.show()
```

Instead of starting with a random noisy latent we can start with an actual image.
This is the Img2Img pipeline.

```python
sd.img_2_img(['Donald Trump'], img, start_step=20, num_inference_steps=50, guidance_scale=3, seed=None)[0]
```

```python

```

```python

```
