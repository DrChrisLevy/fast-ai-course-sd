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

# Stable Diffusion Class

```{code-cell} ipython3
import os
os.chdir('/workspace')
```

```{code-cell} ipython3
from stable_diffusion import StableDiffusion
sd = StableDiffusion()
```

```{code-cell} ipython3
prompt = ["photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details,unreal engine 5, cinematic, masterpiece, art by studio ghibli "]
sd.text_to_img(prompt,num_inference_steps=50,guidance_scale=7.5,seed=498)[0]
```

```{code-cell} ipython3
from fastdownload import FastDownload
from PIL import Image
p = FastDownload().download('https://lafeber.com/pet-birds/wp-content/uploads/2018/06/Scarlet-Macaw-2.jpg')
init_image = Image.open(p).convert("RGB")
init_image.size
init_image.thumbnail((512, 512))
init_image
```

```{code-cell} ipython3
prompt = ["epic photo close up of master chief, 4K, HD, unreal engine"]

sd.img_2_img(prompt, init_image, start_step=20, num_inference_steps=50, guidance_scale=8, seed=2893)[0]
```

```{code-cell} ipython3
prompt = ["Paris in Spring, digital art"]
text_input, text_embeddings = sd.embed_text(prompt)
start_img = sd.embeddings_to_img(text_embeddings, num_inference_steps=30, guidance_scale=7.5, seed=23532)[0]
start_img
```

```{code-cell} ipython3
_, a = sd.embed_text(['Paris in Spring, digital art'])
_, b = sd.embed_text(['Paris in Summer, digital art'])
_, c = sd.embed_text(['Paris in Fall, digital art'])
_, d = sd.embed_text(['Paris in Winter, digital art'])
```

```{code-cell} ipython3
start_img_a = sd.embeddings_to_img(a, num_inference_steps=30, guidance_scale=7.5, seed=23532)[0]
start_img_a
```

```{code-cell} ipython3
start_img_b = sd.embeddings_to_img(b, num_inference_steps=30, guidance_scale=7.5, seed=23532)[0]
start_img_b
```

```{code-cell} ipython3
start_img_c = sd.embeddings_to_img(c, num_inference_steps=30, guidance_scale=7.5, seed=23532)[0]
start_img_c
```

```{code-cell} ipython3
start_img_d = sd.embeddings_to_img(d, num_inference_steps=30, guidance_scale=7.5, seed=23532)[0]
start_img_d
```

```{code-cell} ipython3
import torch
imgs = []
# increase steps for smoother transitions
for w in torch.linspace(0, 1, steps=10):
    imgs.append(sd.embeddings_to_img(torch.lerp(a, b, torch.full_like(a, w)), num_inference_steps=30, guidance_scale=7.5, seed=23532)[0])

for w in torch.linspace(0, 1, steps=10):
    imgs.append(sd.embeddings_to_img(torch.lerp(b, c, torch.full_like(b, w)), num_inference_steps=30, guidance_scale=7.5, seed=23532)[0])

for w in torch.linspace(0, 1, steps=10):
    imgs.append(sd.embeddings_to_img(torch.lerp(c, d, torch.full_like(c, w)), num_inference_steps=30, guidance_scale=7.5, seed=23532)[0])    
```

```{code-cell} ipython3
imgs[0].save("paris_seasons_interp.gif", save_all=True, append_images=imgs[1:], duration=200, loop=0)
```

```{code-cell} ipython3
from IPython.display import Image as IPythonImage
IPythonImage(url='paris_seasons_interp.gif')
```

```{code-cell} ipython3

```
