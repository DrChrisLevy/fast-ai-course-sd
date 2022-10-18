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

# Diffusion Loop Notes

These were some notes I took from the amazing notebook [here](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb). All the code and notes
were copy/pasted from there but then I added some more notes for my learning.

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

```{code-cell} ipython3
# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);
```

## A diffusion loop

```{code-cell} ipython3
# Some settings
prompt = ["Highly detailed small japanese house, radiant light, detailed and intricate environment"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 30            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(1632)   # Seed generator to create the inital latent noise
batch_size = 1
```

The tokenizer converts the text prompt to the tensor of `input_ids`
and it also gives the `attention_mask`. The text prompt is stored as
these sequence of ids of length 77. Depending on the text prompt
and the number of words you may see some padding. This is why the special
token `49407` is repeated a bunch and all the values in the `attention_mask`
are set to 0 for those spots.

```{code-cell} ipython3
# Prep text 
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_input
```

We can feed the `text_input` (ids and attention mask) through the CLIP
text encoder to get the text embeddings which is a tensor with the embeddings.
There are 77 embeddings (one for each token) and the dimension of each embedding is 768.

```{code-cell} ipython3
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```

```{code-cell} ipython3
text_embeddings.shape
```

```{code-cell} ipython3
text_embeddings[0]
```

```{code-cell} ipython3
max_length = text_input.input_ids.shape[-1]
max_length
```

Recall that we had a bunch of padding in the sequence for the text prompt
with the same token input_id repeated. The reason the embeddings are not all 
identical is because of the positional embeddings are used here too.

+++

Now for the unconditional input. Now what is that all about?

**TODO: learn more about this and add it here later**

We are creating another text now which is just an empty string.
Lets look at the input_ids for it. Note that we create the same
length sequence 77 of the text embeddings for the text prompt.

```{code-cell} ipython3
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_input
```

```{code-cell} ipython3
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
uncond_embeddings.shape
```

```{code-cell} ipython3
uncond_embeddings[0] # the token embeddings for the empty string.
```

So we have `text_embeddings` from the text prompt and we have `uncond_embeddings`
from the empty string `""`. But now we are going to concatenate them
into one tensor and rename it `text_embeddings`. So now it is a batch of CLIP
embeddings.

```{code-cell} ipython3
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
text_embeddings.shape
```

So now `text_embeddings` is a batch of size 2
and the embeddings for the text prompt is `text_embeddings[1]`
and the embeddings for the `uncond_input` are `text_embeddings[0]`

+++

Here is the scheduler. There is probably a lot more we could say
but for now just think of it giving us the time steps
and the amount/strength of noise to use as each step.
`num_inference_steps` is typically like 50 or something
but there is some recent research where this can apparently be
a lot shorter. But with this current implementation
we need to use like 30 to 50 steps to get a nice image
in the end.

```{code-cell} ipython3
# Prep Scheduler
scheduler.set_timesteps(num_inference_steps)
```

```{code-cell} ipython3
scheduler.timesteps
```

```{code-cell} ipython3
scheduler.sigmas
```

Why is `scheduler.sigmas` length one more than `scheduler.timesteps` ?

```{code-cell} ipython3
# Plotting this noise schedule:
plt.plot(scheduler.sigmas)
plt.title('Noise Schedule')
plt.xlabel('Sampling step')
plt.ylabel('sigma')
plt.show()
```

We are going to start with a random tensor in the latent space. 
When creating 3x512x512 images then this latent space 
has dimensions 4x64x64. A 48 times reduction in size.
This is what the VAE (variational auto encoder) is for.
We can used the decoder of the VAE to get from the latent space
back to the original image pixel space. 

Remember below in this example `batch_size=1`, `unet.in_channels=4`, `height=512`, `width=512`.

```{code-cell} ipython3
# Prep latents
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(torch_device)
latents.shape
```

Let's visualize the four channels of this random latent representation.
It's just noise.

```{code-cell} ipython3
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    axs[c].imshow(latents[0][c].cpu(), cmap='Greys')
```

At the moment the values come from a normal distribution with mean 0 and variance 1.

```{code-cell} ipython3
import seaborn as sns
print(f'mean {latents[0].cpu().flatten().mean()}', f'std {latents[0].cpu().flatten().std()}')
sns.displot(latents[0].cpu().flatten())
```

Let's grab the first sigma value from the `scheduler`. It could be accessed
as `scheduler.sigmas[0]` or we can use `scheduler.init_noise_sigma`

```{code-cell} ipython3
assert scheduler.init_noise_sigma == scheduler.sigmas[0]
```

```{code-cell} ipython3
scheduler.init_noise_sigma
```

Lets scale the amount of noise in the `latents` by the first sigma factor/value,
`scheduler.init_noise_sigma`.

```{code-cell} ipython3
latents = latents * scheduler.init_noise_sigma
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    axs[c].imshow(latents[0][c].cpu(), cmap='Greys')
```

```{code-cell} ipython3
print(latents[0].cpu().flatten().mean(), latents[0].cpu().flatten().std())
sns.displot(latents[0].cpu().flatten())
```

Okay, so we are starting with some amount of random  noise
from the scheduler within the latent space.

```{code-cell} ipython3
latents.shape
```

```{code-cell} ipython3
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

def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
```

We can look at the latents in the original pixel space too.

```{code-cell} ipython3
# decode the update latents to see whats going on
plt.figure()
plt.title('latents in pixel space')
plt.imshow(latents_to_pil(latents)[0])
```

```{code-cell} ipython3
# Loop
with autocast("cuda"):
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
        
        # decode the predicted noise so we can see whats going on at each step.
        # Just for learning purposes. Not need to generate the final image.
        plt.figure()
        plt.subplot(1, 2, 1)
        if i == 0:
            plt.title(f'LEFT: latent in pixel space , RIGHT: predicted noise in latent space {i,float(t),float(sigma)}')
        plt.imshow(latents_to_pil(latents)[0])
        # decode the update latents to see whats going on
        plt.subplot(1, 2, 2)
        plt.imshow(latents_to_pil(noise_pred.type(torch.float32))[0])

```
