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

```

```{code-cell} ipython3
noises1 = []
for i in range(10):
    _, noise_pred1 = sd.add_noise_and_predict_one_step(["a horse"], input_image, 50, 30, 10, seed=(i+1)*100)
    noises1.append(noise_pred1[0])
noises1 = torch.stack(noises1)
```

```{code-cell} ipython3
noises2 = []
for i in range(10):
    _, noise_pred2 = sd.add_noise_and_predict_one_step(["a zebra"], input_image, 50, 30, 10, seed=(i+1)*100)
    noises2.append(noise_pred2[0]) 
noises2 = torch.stack(noises2)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
noises1 = torch.clamp(noises1, torch.mean(noises1) - 3 * torch.std(noises1), torch.mean(noises1) + 3 * torch.std(noises1))
noises2 = torch.clamp(noises2,  torch.mean(noises2) - 3 * torch.std(noises2), torch.mean(noises2) + 3 * torch.std(noises2))

diffs1 = noises1 - noises2
diffs2 = noises2 - noises1

diffs1 = torch.mean(diffs1, axis=0)[None,:]
diffs2 = torch.mean(diffs2, axis=0)[None,:]


diffs1  = (diffs1 - diffs1.min())/(diffs1.max() - diffs1.min())
diffs2  = (diffs2 - diffs2.min())/(diffs2.max() - diffs2.min())

diffs1 = diffs1[0] 
diffs2 = diffs2[0] 

diffs1 = torch.where(diffs1 > 0.6,1.,0.)
diffs2 = torch.where(diffs2 > 0.52,1.,0.)

diffs1 = torch.max(diffs1,dim=0)[0]
diffs2 = torch.max(diffs2,dim=0)[0]
```

```{code-cell} ipython3
plt.imshow(diffs1.cpu())
```

```{code-cell} ipython3
plt.imshow(diffs2.cpu())
```

```{code-cell} ipython3
MASK = np.maximum(diffs1.cpu().numpy(), diffs2.cpu().numpy())
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
MASK = (np.array(Image.fromarray(MASK*255).convert('RGB').resize((512,512)))/255.)
plt.imshow(MASK)
```

```{code-cell} ipython3
new_img = sd.img_2_img(['a zebra'], input_image, start_step=30, num_inference_steps=50, seed=42)[0]
new_img
```

```{code-cell} ipython3
Image.fromarray((input_image*(1-MASK) + MASK*np.array(new_img)).astype('uint8'))
```

```{code-cell} ipython3

```
