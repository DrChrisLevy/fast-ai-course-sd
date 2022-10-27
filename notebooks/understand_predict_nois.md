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
tokenizer = sd.tokenizer
text_encoder = sd.text_encoder
```

```{code-cell} ipython3
!curl --output horse.jpg 'https://nationaltoday.com/wp-content/uploads/2022/02/International-Zebra-Day-640x514.jpg'
```

```{code-cell} ipython3
# Load the image with PIL
input_image = Image.open('horse.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
def one_step(prompt = ["a zebra"], seed=42, sampling_step = 46, guidance_scale=7.5):
    latents = sd.pil_to_latent(input_image)
    sd.scheduler.set_timesteps(50)
    noise = torch.randn_like(latents) # Random noise
    generator = torch.manual_seed(seed) # this is important !!
    latents = sd.scheduler.add_noise(latents, noise, timesteps=torch.tensor([sd.scheduler.timesteps[sampling_step]]))
    latents = latents.to(torch_device).float()
    sd.latents_to_pil(latents.float())[0] # Display
    
    sigma = scheduler.sigmas[sampling_step]
    t = sd.scheduler.timesteps[sampling_step]
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
noises = []
for seed in range(10):
    noise = one_step(prompt = ["a zebra"], guidance_scale=10, seed=seed, sampling_step=20)
    noises.append(noise[0].cpu().numpy())
```

```{code-cell} ipython3
noise_pred1 = np.mean(np.array(noises),axis=0)[None,]
```

```{code-cell} ipython3
noises = []
for seed in range(10):
    noise = one_step(prompt = ["a horse"], guidance_scale=10, seed=seed, sampling_step=20)
    noises.append(noise[0].cpu().numpy())
```

```{code-cell} ipython3
noise_pred2 = np.mean(np.array(noises),axis=0)[None,:]
```

```{code-cell} ipython3

```

```{code-cell} ipython3
contrast = torch.tensor(noise_pred1-noise_pred2).to(torch_device)
cls = StableDiffusion
```

```{code-cell} ipython3
img = sd.latents_to_pil(contrast)[0]
img
```

```{code-cell} ipython3

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

```{code-cell} ipython3
X = np.array(img)
```

```{code-cell} ipython3
plt.imshow(X)
```

```{code-cell} ipython3
from PIL import Image
import numpy as np

col = img
gray = col.convert('L')

# Let numpy do the heavy lifting for converting pixels to pure black or white
bw = np.asarray(gray).copy()

# Pixel range is 0...255, 256/2 = 128
bw[bw < 128] = 0    # Black
bw[bw >= 128] = 255 # White

# Now we put it back in Pillow/PIL land
imfile = Image.fromarray(bw)
imfile
```

```{code-cell} ipython3
def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv
```

```{code-cell} ipython3
def binarize_this(X, thresh_val=127, with_plot=False, gray_scale=False):
    image_src = X
    if not gray_scale:
        cmap_val = None
        r_img, g_img, b_img = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
        
        r_b = convert_binary(image_matrix=r_img, thresh_val=thresh_val)
        g_b = convert_binary(image_matrix=g_img, thresh_val=thresh_val)
        b_b = convert_binary(image_matrix=b_img, thresh_val=thresh_val)
        
        image_b = np.dstack(tup=(r_b, g_b, b_b))
    else:
        cmap_val = 'gray'
        image_b = convert_binary(image_matrix=image_src, thresh_val=thresh_val)
    
    if with_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Binarized")
        
        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        return True
    return image_b
```

```{code-cell} ipython3
binarize_this(X,with_plot=True,gray_scale=True)
```

```{code-cell} ipython3
sd.img_2_img(['a zebra'], input_image, start_step=40, num_inference_steps=50, guidance_scale=4, seed=42)[0]
```

```{code-cell} ipython3
sd.img_2_img(['a cow'], input_image, start_step=30, num_inference_steps=50, guidance_scale=4, seed=42)[0]
```

```{code-cell} ipython3

```
