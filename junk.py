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

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


class StableDiffusion:
    vae = None
    tokenizer = None
    text_encoder = None
    unet = None
    scheduler = None

    height = 512
    width = 512

    def __init__(self):
        if StableDiffusion.vae is None:
            StableDiffusion.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="vae"
            ).to(torch_device)
            StableDiffusion.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            StableDiffusion.unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="unet"
            ).to(torch_device)
            StableDiffusion.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(torch_device)
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
            text_embeddings = cls.text_encoder(text_input.input_ids.to(torch_device))[0]
        return text_input, text_embeddings

    @classmethod
    def text_to_img(cls, prompt, num_inference_steps=30, guidance_scale=7.5, seed=None):
        """
        23532 for seed for tests
        """
        batch_size = len(prompt)
        generator = torch.manual_seed(seed) if seed else None

        text_input, text_embeddings = cls.embed_text(prompt)
        uncond_input, uncond_embeddings = cls.embed_text([""] * batch_size)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # Prep Scheduler
        cls.scheduler.set_timesteps(num_inference_steps)

        # Prep latents
        latents = torch.randn(
            (batch_size, cls.unet.in_channels, cls.height // 8, cls.width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * cls.scheduler.init_noise_sigma

        # Loop
        with autocast("cuda"):
            for i, t in tqdm(enumerate(cls.scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = cls.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = cls.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = cls.scheduler.step(noise_pred, t, latents).prev_sample

        return cls.latents_to_pil(latents)

    @classmethod
    def embeddings_to_img(
        cls, text_embeddings, num_inference_steps=30, guidance_scale=7.5, seed=None
    ):
        batch_size = text_embeddings.shape[0]
        generator = torch.manual_seed(seed) if seed else None

        uncond_input, uncond_embeddings = cls.embed_text([""] * batch_size)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        # Prep Scheduler
        cls.scheduler.set_timesteps(num_inference_steps)

        # Prep latents
        latents = torch.randn(
            (batch_size, cls.unet.in_channels, cls.height // 8, cls.width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)
        latents = latents * cls.scheduler.init_noise_sigma

        # Loop
        with autocast("cuda"):
            for i, t in tqdm(enumerate(cls.scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = cls.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = cls.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = cls.scheduler.step(noise_pred, t, latents).prev_sample

        return cls.latents_to_pil(latents)

    @classmethod
    def img_2_img(
        cls, prompt, image, start_step=10, num_inference_steps=50, guidance_scale=8, seed=32
    ):
        encoded = cls.pil_to_latent(image)

        torch.manual_seed(seed)
        batch_size = len(prompt)
        text_input, text_embeddings = cls.embed_text(prompt)
        uncond_input, uncond_embeddings = cls.embed_text([""] * batch_size)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler (setting the number of inference steps)
        cls.scheduler.set_timesteps(num_inference_steps)

        # Prep latents (noising appropriately for start_step)
        noise = torch.randn_like(encoded)
        latents = cls.scheduler.add_noise(
            encoded, noise, timesteps=torch.tensor([cls.scheduler.timesteps[start_step]])
        )
        latents = latents.to(torch_device).float()

        # Loop
        for i, t in tqdm(enumerate(cls.scheduler.timesteps)):
            if i > start_step:  # << This is the only modification to the loop we do

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = cls.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = cls.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = cls.scheduler.step(noise_pred, t, latents).prev_sample

        return cls.latents_to_pil(latents)

    @classmethod
    def pil_to_latent(cls, input_im):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latent = cls.vae.encode(
                tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device) * 2 - 1
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
