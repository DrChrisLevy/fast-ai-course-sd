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
!curl --output horse.jpg 'https://th-thumbnailer.cdn-si-edu.com/aZINl-wLtWrRfYD9ni4WU3STuDg=/fit-in/1600x0/filters:focal(3008x2005:3009x2006)/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer_public/6b/c3/6bc305cb-95dd-4e22-b45b-108c6e218200/gettyimages-1144833913.jpg'
```

```{code-cell} ipython3
# Load the image with PIL
input_image = Image.open('horse.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
# visualize adding noise to the image
sd.latents_to_pil(sd.add_noise_to_image(input_image, 50, 40, seed=42))[0]
```

```{code-cell} ipython3
num_inference_steps = 50
def one_step(prompt = ["a horse"], seed=42, sampling_step = 20, guidance_scale=7.5):
    # Prep text
    text_input, text_embeddings = sd.embed_text(prompt)
    batch_size = text_embeddings.shape[0]
    uncond_input, uncond_embeddings = sd.embed_text([""] * batch_size)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = sd.add_noise_to_image(input_image, num_inference_steps, sampling_step, seed)
    
    t = sd.scheduler.timesteps[sampling_step]
    
    latents, noise_pred = sd.diffusion_step(latents, text_embeddings, t, guidance_scale)
    
    return noise_pred
```

```{code-cell} ipython3
noises1 = []
for i in range(10):
    noise = one_step(prompt = ["a horse"], guidance_scale=10, seed=i*100, sampling_step=25)
    noises1.append(noise[0])
noises1 = torch.stack(noises1)
```

```{code-cell} ipython3
noises2 = []
for i in range(10):
    noise = one_step(prompt = ["a zebra"], guidance_scale=10, seed=i*100, sampling_step=25)
    noises2.append(noise[0]) 
noises2 = torch.stack(noises2)
```

```{code-cell} ipython3
diffs1 = [np.array(sd.latents_to_pil(n1[None,:] - n2[None,:])[0]) for n1,n2 in zip(noises1,noises2)]
diffs2 = [np.array(sd.latents_to_pil(n2[None,:] - n1[None,:])[0]) for n1,n2 in zip(noises1,noises2)]
```

```{code-cell} ipython3
X1 = np.mean(np.array(diffs1),axis=0).astype('uint8')
plt.imshow(X1)
```

```{code-cell} ipython3
# convert to 1 channel images
X1 = np.mean(X1,axis=2)
plt.imshow(X1)
```

```{code-cell} ipython3
X2 = np.mean(np.array(diffs2),axis=0).astype('uint8')
X2 = np.mean(X2,axis=2)
plt.imshow(X2)
```

```{code-cell} ipython3
X1B = ((X1-X1.min())/(X1.max()-X1.min()) < 0.5).astype('uint8')
plt.imshow(X1B.astype('uint8'))
```

```{code-cell} ipython3
X2B = ((X2-X2.min())/(X2.max()-X2.min()) < 0.32).astype('uint8')
plt.imshow(X2B)
```

```{code-cell} ipython3
MASK = np.maximum(X1B,X2B).astype('uint8')
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
MASK = np.array([MASK,MASK,MASK]).transpose(1,2,0)
```

```{code-cell} ipython3
MASK.shape
```

```{code-cell} ipython3
input_image
```

```{code-cell} ipython3
new_img = sd.img_2_img(['a zebra'], input_image,start_step=25,num_inference_steps=50, seed=50)[0]
new_img
```

```{code-cell} ipython3
Image.fromarray(input_image*(1-MASK) + MASK*np.array(new_img))
```

```{code-cell} ipython3
input_image
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
