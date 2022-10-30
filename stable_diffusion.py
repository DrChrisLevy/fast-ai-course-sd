import torch
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from torchvision import transforms as tfms


# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()


class StableDiffusion:
    vae = None
    tokenizer = None
    text_encoder = None
    unet = None
    scheduler = None
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    height = 512
    width = 512

    def __init__(self):
        if StableDiffusion.vae is None:
            StableDiffusion.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="vae"
            ).to(StableDiffusion.torch_device)
            StableDiffusion.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            StableDiffusion.unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="unet"
            ).to(StableDiffusion.torch_device)
            StableDiffusion.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(StableDiffusion.torch_device)
            StableDiffusion.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )

    @classmethod
    def embed_text(cls, text):
        max_length = cls.tokenizer.model_max_length
        text_input = cls.tokenizer(
            text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = cls.text_encoder(text_input.input_ids.to(cls.torch_device))[0]
        return text_input, text_embeddings

    @classmethod
    def add_noise_to_latents(cls, latents, num_inference_steps=50, sampling_step=40, seed=42):
        cls.scheduler.set_timesteps(num_inference_steps)
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn_like(latents)
        latents = cls.scheduler.add_noise(
            latents, noise, timesteps=torch.tensor([cls.scheduler.timesteps[sampling_step]])
        )
        latents = latents.to(cls.torch_device).float()
        return latents

    @classmethod
    def add_noise_to_image(cls, image, num_inference_steps=50, sampling_step=40, seed=42):
        latents = cls.pil_to_latent(image)
        return cls.add_noise_to_latents(latents, num_inference_steps, sampling_step, seed)

    @classmethod
    def diffusion_step(cls, latents, text_embeddings, t, guidance_scale):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = cls.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = cls.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = cls.scheduler.step(noise_pred, t, latents).prev_sample
        return latents, noise_pred

    @classmethod
    def diffusion_loop(cls, text_embeddings, num_inference_steps=30, guidance_scale=7.5, seed=None):
        batch_size = text_embeddings.shape[0]
        uncond_input, uncond_embeddings = cls.embed_text([""] * batch_size)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # start with noisy random latent
        latents = torch.zeros((batch_size, 4, cls.height // 8, cls.width // 8))
        cls.scheduler.set_timesteps(num_inference_steps)
        latents = cls.add_noise_to_latents(latents, num_inference_steps, 0, seed)

        # Loop
        with autocast("cuda"):
            for i, t in tqdm(enumerate(cls.scheduler.timesteps)):
                latents, _ = cls.diffusion_step(latents, text_embeddings, t, guidance_scale)

        return cls.latents_to_pil(latents)

    @classmethod
    def text_to_img(cls, prompt, num_inference_steps=30, guidance_scale=7.5, seed=None):
        text_input, text_embeddings = cls.embed_text(prompt)
        return cls.diffusion_loop(text_embeddings, num_inference_steps, guidance_scale, seed)

    @classmethod
    def embeddings_to_img(
        cls, text_embeddings, num_inference_steps=30, guidance_scale=7.5, seed=None
    ):
        return cls.diffusion_loop(text_embeddings, num_inference_steps, guidance_scale, seed)

    @classmethod
    def img_2_img(
        cls, prompt, image, start_step=10, num_inference_steps=50, guidance_scale=8, seed=None,
    ):

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

        # Loop
        with autocast("cuda"):
            for i, t in tqdm(enumerate(cls.scheduler.timesteps)):
                if i > start_step:  # << This is the only modification to the loop we do
                    latents, _ = cls.diffusion_step(latents, text_embeddings, t, guidance_scale)

        return cls.latents_to_pil(latents)

    @classmethod
    def pil_to_latent(cls, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = cls.vae.encode(
                tfms.ToTensor()(input_im).unsqueeze(0).to(cls.torch_device) * 2 - 1
            )  # Note scaling
        return 0.18215 * latent.latent_dist.sample()

    @classmethod
    def latents_to_pil(cls, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = cls.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
