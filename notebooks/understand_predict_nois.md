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
latents = sd.add_noise_to_image(input_image, 50, 30, seed=42)
sd.latents_to_pil(latents)[0]
```

```{code-cell} ipython3
# going back in reverse direction with one diffusion step
_, text_embeddings = sd.embed_text("a horse")
batch_size = text_embeddings.shape[0]
uncond_input, uncond_embeddings = sd.embed_text([""] * batch_size)
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents1, noise_pred1 = sd.diffusion_step(latents, text_embeddings, sd.scheduler.timesteps[30], 8)
sd.latents_to_pil(latents1)[0]
```

```{code-cell} ipython3
cls = StableDiffusion
def add_noise_and_predict_one_step(
    prompt, image, num_inference_steps=50, sampling_step=30, guidance_scale=8, seed=42
):
    # add noise
    latents = cls.add_noise_to_image(image, num_inference_steps, sampling_step, seed)

    # denoise for one step
    _, text_embeddings = cls.embed_text(prompt)
    batch_size = text_embeddings.shape[0]
    uncond_input, uncond_embeddings = cls.embed_text([""] * batch_size)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents, noise_pred = cls.diffusion_step(
        latents, text_embeddings, cls.scheduler.timesteps[sampling_step], guidance_scale
    )
    return latents, noise_pred
```

```{code-cell} ipython3
latents, noise_pred = add_noise_and_predict_one_step( ['a horse'],input_image, num_inference_steps=50, sampling_step=30, guidance_scale=8, seed=42)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
cls = StableDiffusion

def img_2_img(prompt, image, start_step=30, num_inference_steps=50, guidance_scale=8, seed=42):

    batch_size = 1  # only supported
    if isinstance(prompt, torch.Tensor):
        assert prompt.shape == torch.Size([1, 77, 768])  # TODO
        text_embeddings = prompt
    elif isinstance(prompt, list):
        assert len(prompt) == 1
        _, text_embeddings = cls.embed_text(prompt)
    else:
        raise Exception("prompt not proper format")

    uncond_input, uncond_embeddings = cls.embed_text([""] * batch_size)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = cls.add_noise_to_image(image, num_inference_steps, start_step, seed)

    noise_preds = torch.tensor([], device='cuda')
    with autocast("cuda"):
        for i, t in tqdm(enumerate(cls.scheduler.timesteps)):
            if i > start_step:  # << This is the only modification to the loop we do
                latents, noise_pred = cls.diffusion_step(latents, text_embeddings, t, guidance_scale)
                noise_preds = torch.concat((noise_preds, noise_pred))

    return cls.latents_to_pil(latents), noise_preds
```

```{code-cell} ipython3
noises1 = []
for i in range(10):
    vis, noise_preds1 = img_2_img( ["a horse"], input_image, 30, 50, 10, (i+1)*100)
    noises1.append(noise_preds1[0]) # TODO what to do with the other noises
noises1 = torch.stack(noises1)
```

```{code-cell} ipython3
noises2 = []
for i in range(10):
    vis, noise_preds2 = img_2_img( ["a zebra"], input_image, 30, 50, 10, (i+1)*100)
    noises2.append(noise_preds2[0]) # TODO what to do with the other noises
noises2 = torch.stack(noises2)
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

diffs1 = diffs1[0] # torch.Size([4, 64, 64])
diffs2 = diffs2[0] # torch.Size([4, 64, 64])

diffs1 = torch.where(diffs1 > 0.67,1.,0.)
diffs2 = torch.where(diffs2 > 0.5,1.,0.)

diffs1 = torch.max(diffs1,dim=0)[0]
diffs2 = torch.max(diffs2,dim=0)[0]
```

```{code-cell} ipython3
plt.imshow(diffs2.cpu())
```

```{code-cell} ipython3
plt.imshow(diffs1.cpu())
```

```{code-cell} ipython3
MASK = np.maximum(diffs1.cpu().numpy(), diffs2.cpu().numpy())
```

```{code-cell} ipython3
plt.imshow(MASK)
```

```{code-cell} ipython3
MASK = np.array(Image.fromarray(MASK*255).convert('RGB').resize((512,512)))/255.
plt.imshow(MASK)
```

```{code-cell} ipython3
mask = diffs2.cpu().numpy()
mask = Image.fromarray(np.array([mask,mask,mask]).reshape(64,64,3).astype('uint8')).resize((512,512))
```

```{code-cell} ipython3
mask
```

```{code-cell} ipython3

```

```{code-cell} ipython3
Image.fromarray(mask)
```

```{code-cell} ipython3
Image.fromarray(diffs2.cpu().numpy().resize((512,512)).astype('uint8'))
```

```{code-cell} ipython3
diffs2.cpu().numpy().copy().resize((512,512)).asty
```

```{code-cell} ipython3
diffs2.cpu().numpy().resize((512,512))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
Image.fromarray(diffs2.cpu().numpy())
```

```{code-cell} ipython3
Image.fromarray(diffs2.cpu().numpy()).resize((512, 512))
```

```{code-cell} ipython3

```

```{code-cell} ipython3
mask_array = latent_mask.max(1).values.detach().cpu().reshape(64,64).numpy()
mask = Image.fromarray(mask_array).resize((512, 512))
mask_inv = Image.fromarray(~mask_array).resize((512, 512))
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
diffs1 = [np.array(sd.latents_to_pil(n1[None,:] - n2[None,:])[0]) for n1,n2 in zip(noises1,noises2)]
diffs2 = [np.array(sd.latents_to_pil(n2[None,:] - n1[None,:])[0]) for n1,n2 in zip(noises1,noises2)]
```

```{code-cell} ipython3
import seaborn as sns
sns.distplot(noises2[0].flatten().cpu())
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
X1 = np.mean(np.array(diffs1),axis=0).astype('uint8')
plt.imshow(X1)
```

```{code-cell} ipython3
X1.shape
```

```{code-cell} ipython3

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
new_img = sd.img_2_img(['a zebra'], input_image, start_step=30,num_inference_steps=50, seed=42)[0]
new_img
```

```{code-cell} ipython3
Image.fromarray(input_image*(1-MASK) + MASK*np.array(new_img))
```

```{code-cell} ipython3

```
