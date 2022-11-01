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
def diff_edit(input_image, prompt1, prompt2, thresh1, thresh2):
    
    noises1 = []
    for i in range(5):
        _, noise_pred1 = sd.add_noise_and_predict_one_step(
            prompt1, input_image, 50, 30, 5, seed=(i + 1) * 100
        )
        noises1.append(noise_pred1[0])
    noises1 = torch.stack(noises1)

    noises2 = []
    for i in range(5):
        _, noise_pred2 = sd.add_noise_and_predict_one_step(
            prompt2, input_image, 50, 30, 5, seed=(i + 1) * 100
        )
        noises2.append(noise_pred2[0])
    noises2 = torch.stack(noises2)

    noises1 = torch.clamp(
        noises1,
        torch.mean(noises1) - 2 * torch.std(noises1),
        torch.mean(noises1) + 2 * torch.std(noises1),
    )
    noises2 = torch.clamp(
        noises2,
        torch.mean(noises2) - 2 * torch.std(noises2),
        torch.mean(noises2) + 2 * torch.std(noises2),
    )

    diffs1 = noises1 - noises2
    diffs2 = noises2 - noises1

    diffs1 = torch.mean(diffs1, axis=0)[None, :]
    diffs2 = torch.mean(diffs2, axis=0)[None, :]

    diffs1 = (diffs1 - diffs1.min()) / (diffs1.max() - diffs1.min())
    diffs2 = (diffs2 - diffs2.min()) / (diffs2.max() - diffs2.min())

    diffs1 = diffs1[0]
    diffs2 = diffs2[0]

    diffs1 = torch.where(diffs1 > thresh1, 1.0, 0.0)
    diffs2 = torch.where(diffs2 > thresh2, 1.0, 0.0)

    diffs1 = torch.max(diffs1, dim=0)[0]
    diffs2 = torch.max(diffs2, dim=0)[0]
    
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(diffs1.cpu())
    axarr[1].imshow(diffs2.cpu())
    
    MASK = np.maximum(diffs1.cpu().numpy(), diffs2.cpu().numpy())
    MASK = (np.array(Image.fromarray(MASK*255).convert('RGB').resize((512,512)))/255.)
    return MASK 
```

## Horse 2 Zebra

```{code-cell} ipython3
!curl --output input_image.jpg 'https://th-thumbnailer.cdn-si-edu.com/aZINl-wLtWrRfYD9ni4WU3STuDg=/fit-in/1600x0/filters:focal(3008x2005:3009x2006)/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer_public/6b/c3/6bc305cb-95dd-4e22-b45b-108c6e218200/gettyimages-1144833913.jpg'
```

```{code-cell} ipython3
input_image = Image.open('input_image.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
# Load the image with PIL
MASK = diff_edit(input_image, ["a horse"], ["a zebra"], thresh1=0.62, thresh2=0.55)
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
new_img = sd.img_2_img(['a zebra'], input_image, start_step=30, num_inference_steps=50, seed=1, guidance_scale=5)[0]
Image.fromarray((input_image*(1-MASK) + MASK*np.array(new_img)).astype('uint8'))
```

## Dog 2 Dog


```{code-cell} ipython3
!curl --output input_image.jpg 'https://c8.alamy.com/zooms/9/838ca3778356412f868cc9e154a86049/2dkb0gp.jpg'
```

```{code-cell} ipython3
input_image = Image.open('input_image.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
MASK = diff_edit(input_image, ["a dalmation dog running in a field"], 
                              ["a black lab dog running in a field"], 
                               thresh1=0.53, thresh2=0.64)
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
new_img = sd.img_2_img(["a black lab dog running in a field"], input_image, 
                       start_step=18, num_inference_steps=50, seed=10, 
                       guidance_scale=5)[0]
Image.fromarray((input_image*(1-MASK) + MASK*np.array(new_img)).astype('uint8'))
```

## Bird 2 Bird


```{code-cell} ipython3
!curl --output input_image.jpg 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Snowy_Owl_%28240866707%29.jpeg'
```

```{code-cell} ipython3
input_image = Image.open('input_image.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
MASK = diff_edit(input_image, ["owl"], 
                              ["eagle"], 
                               thresh1=0.5, thresh2=0.6)
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
new_img = sd.img_2_img(["eagle"], input_image, 
                       start_step=20, num_inference_steps=50, seed=30, 
                       guidance_scale=5)[0]
Image.fromarray((input_image*(1-MASK) + MASK*np.array(new_img)).astype('uint8'))
```

## Jesse 2 Walter

```{code-cell} ipython3
!curl --output input_image.jpg 'https://cdn.wionews.com/sites/default/files/2022/08/27/290896-aaron-paul-jesse-pinkman.PNG'
input_image = Image.open('input_image.jpg').resize((512, 512))
input_image
```

```{code-cell} ipython3
MASK = diff_edit(input_image, ["jesse pinkman standing in a field"], 
                              ["walter white standing in a field"], 
                               thresh1=0.59, thresh2=0.47)
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
new_img = sd.img_2_img(["walter white face standing in a field wearing a black tshirt with a skeleton"], input_image, 
                       start_step=39, num_inference_steps=100, seed=10, 
                       guidance_scale=8)[0]
Image.fromarray((input_image*(1-MASK) + MASK*np.array(new_img)).astype('uint8'))
```
