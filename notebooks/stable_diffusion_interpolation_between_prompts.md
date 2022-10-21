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

# Stable Diffusion Interpolation Between Prompts

```{code-cell} ipython3
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms as tfms

# For video display:
from IPython.display import HTML
from base64 import b64encode

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Loading the models

This code (and that in the next section) comes from the [Huggingface example notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb). 

This will download and set up the relevant models and components we'll be using. Let's just run this for now and move on to the next section to check that it all works before diving deeper.

If you've loaded a pipeline, you can also access these components using `pipe.unet`, `pipe.vae` and so on.

```{code-cell} ipython3
def load_models():
    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    # The noise scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    return vae, text_encoder, unet, tokenizer, scheduler
```

```{code-cell} ipython3
vae, text_encoder, unet, tokenizer, scheduler = load_models()
```

```{code-cell} ipython3
torch.cuda.empty_cache()
```

When you keep creating objects on cpu overwriting previous object the memory
typically does not increase. But when you you `to('cuda')`, even if the thing is already
there, it keeps increasing the memory. Be careful of that!
When you move an object from gpu to cpu and call `torch.cuda.empty_cache()`
then the memory on gpu will be lowered.

```{code-cell} ipython3
# To the GPU we go!
vae = vae.to(torch_device) if vae.device != torch_device else vae
text_encoder = text_encoder.to(torch_device) if text_encoder.device != torch_device else text_encoder
unet = unet.to(torch_device) if unet.device != torch_device else unet
```



```{code-cell} ipython3
def embed_text(text, max_length=None):
    if max_length is None:
        max_length = tokenizer.model_max_length
    text_input = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    return text_input, text_embeddings

def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


# TODO: Refactor Diffusion Loop in both functions

def embeddings_to_img(prompt_embedding, uncond_embeddings, guidance_scale=7.5, num_inference_steps=30):

    text_embeddings = torch.cat([uncond_embeddings, prompt_embedding])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
      (batch_size, unet.in_channels, height // 8, width // 8),
      generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

    # Loop
    with autocast("cuda"): # this does mixed precision right? Check the speed with
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            # Scale the latents (preconditioning):
            # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    return latents_to_pil(latents)[0]

def img_to_img(image, prompt_embedding, uncond_embeddings, guidance_scale=7.5, num_inference_steps=50, start_step=10):

    text_embeddings = torch.cat([uncond_embeddings, prompt_embedding])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = pil_to_latent(image)
    # Prep latents (noising appropriately for start_step)
    start_sigma = scheduler.sigmas[start_step]
    noise = torch.randn_like(latents)
    latents = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
    latents = latents.to(torch_device).float()

    # Loop
    with autocast("cuda"): # this does mixed precision right? Check the speed with
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            if i < start_step:
                continue
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            # Scale the latents (preconditioning):
            # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    return latents_to_pil(latents)[0]
```

```{code-cell} ipython3
from fastdownload import FastDownload
p = FastDownload().download('https://lafeber.com/pet-birds/wp-content/uploads/2018/06/Scarlet-Macaw-2.jpg')
init_image = Image.open(p).convert("RGB")
init_image.size
init_image.thumbnail((512, 512))
init_image
```

```{code-cell} ipython3
# Some settings
prompt = ["gandalf"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
batch_size = 1

# Prep text 
text_input, prompt_embedding = embed_text(prompt, max_length=tokenizer.model_max_length)
max_length = text_input.input_ids.shape[-1]

# empty string
uncond_input, uncond_embeddings = embed_text([""] * batch_size, max_length)

img_to_img(init_image, prompt_embedding, uncond_embeddings, start_step=20, guidance_scale=7.5)
```

```{code-cell} ipython3
# Some settings
prompt = ["Paris in Spring, digital art"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
batch_size = 1

# Prep text 
text_input, prompt_embedding = embed_text(prompt,  max_length=tokenizer.model_max_length)
max_length = text_input.input_ids.shape[-1]

# empty string
uncond_input, uncond_embeddings = embed_text([""] * batch_size, max_length)

# set seed and generate image
generator = torch.manual_seed(23532)   # Seed generator to create the inital latent noise
start_img = embeddings_to_img(prompt_embedding, uncond_embeddings)
```

```{code-cell} ipython3
start_img
```

```{code-cell} ipython3
_, a = embed_text(['Paris in Spring, digital art'],  max_length=tokenizer.model_max_length)
_, b = embed_text(['Paris in Summer, digital art'],  max_length=tokenizer.model_max_length)
_, c = embed_text(['Paris in Fall, digital art'],  max_length=tokenizer.model_max_length)
_, d = embed_text(['Paris in Winter, digital art'],  max_length=tokenizer.model_max_length)
```

```{code-cell} ipython3
generator = torch.manual_seed(23532)
start_img_a = embeddings_to_img(a, uncond_embeddings)
start_img_a
```

```{code-cell} ipython3
generator = torch.manual_seed(23532)
start_img_b = embeddings_to_img(b, uncond_embeddings)
start_img_b
```

```{code-cell} ipython3
generator = torch.manual_seed(23532)
start_img_c = embeddings_to_img(c, uncond_embeddings)
start_img_c
```

```{code-cell} ipython3
generator = torch.manual_seed(23532)
start_img_d = embeddings_to_img(d, uncond_embeddings)
start_img_d
```

```{code-cell} ipython3
imgs = []

for w in torch.linspace(0, 1, steps=10):
    generator = torch.manual_seed(23532)
    imgs.append(embeddings_to_img(torch.lerp(a, b, torch.full_like(a, w)), uncond_embeddings, num_inference_steps=30))

for w in torch.linspace(0, 1, steps=10):
    generator = torch.manual_seed(23532)
    imgs.append(embeddings_to_img(torch.lerp(b, c, torch.full_like(b, w)), uncond_embeddings, num_inference_steps=30))
    
for w in torch.linspace(0, 1, steps=10):
    generator = torch.manual_seed(23532)
    imgs.append(embeddings_to_img(torch.lerp(c, d, torch.full_like(c, w)), uncond_embeddings, num_inference_steps=30))
    
    
    
```

```{code-cell} ipython3
len(imgs)
```

```{code-cell} ipython3
imgs[0].save("pairs_seasons_100.gif", save_all=True, append_images=imgs[1:], duration=200, loop=0)
```

```{code-cell} ipython3
from IPython.display import Image as IPythonImage
IPythonImage(url='pairs_seasons_100.gif')
```

Above we walked in the latent space from a to b to c to d
but started with a random latent each time. Now
we are going to use the previous image as the input.

```{code-cell} ipython3
imgs = []

img = start_img_a
generator = torch.manual_seed(23532)
for w in torch.linspace(0, 1, steps=10):
    prompt_embedding = torch.lerp(a, b, torch.full_like(a, w))
    img = img_to_img(img, prompt_embedding, uncond_embeddings, guidance_scale=7.5, num_inference_steps=30, start_step=20)
    imgs.append(img)

img = start_img_b
generator = torch.manual_seed(23532)
for w in torch.linspace(0, 1, steps=10):
    prompt_embedding = torch.lerp(b, c, torch.full_like(b, w))
    img = img_to_img(img, prompt_embedding, uncond_embeddings, guidance_scale=7.5, num_inference_steps=30, start_step=20)
    imgs.append(img)
    
img = start_img_c
generator = torch.manual_seed(23532)
for w in torch.linspace(0, 1, steps=10):
    prompt_embedding = torch.lerp(c, d, torch.full_like(c, w))
    img = img_to_img(img, prompt_embedding, uncond_embeddings, guidance_scale=7.5, num_inference_steps=30, start_step=20)
    imgs.append(img)
    
img = start_img_d
generator = torch.manual_seed(23532)
for w in torch.linspace(0, 1, steps=10):
    prompt_embedding = torch.lerp(d, a, torch.full_like(d, w))
    img = img_to_img(img, prompt_embedding, uncond_embeddings, guidance_scale=7.5, num_inference_steps=30, start_step=20)
    imgs.append(img)

                                         
                                         
```

```{code-cell} ipython3
imgs[0].save("pairs_seasons_smooth20.gif", save_all=True, append_images=imgs[1:], duration=200, loop=0)
```

```{code-cell} ipython3
from IPython.display import Image as IPythonImage
IPythonImage(url='pairs_seasons_smooth20.gif')
```

```{code-cell} ipython3
imgs[-1]
```

```{code-cell} ipython3
len(imgs)
```

```{code-cell} ipython3

```
