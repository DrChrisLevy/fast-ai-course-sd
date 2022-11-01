from stable_diffusion import StableDiffusion
import torch


def test_add_noise_step_0():
    sd = StableDiffusion()
    num_inference_steps = 50
    sd.scheduler.set_timesteps(num_inference_steps)
    batch_size = 1

    # was the method in the notebook lesson 9B using sd.scheduler.init_noise_sigma
    seed = 42
    generator = torch.manual_seed(seed)
    latents = torch.randn(
        (batch_size, sd.unet.in_channels, sd.height // 8, sd.width // 8), generator=generator,
    )
    latents = latents.to(StableDiffusion.torch_device)
    latents = latents * sd.scheduler.init_noise_sigma
    latents1 = latents

    # using sd.add_noise_to_latents with sampling_step = 0
    batch_size = 1
    latents = torch.zeros((batch_size, 4, sd.height // 8, sd.width // 8))
    sd.scheduler.set_timesteps(50)
    latents = sd.add_noise_to_latents(latents, num_inference_steps, 0, seed)
    latents2 = latents

    torch.testing.assert_close(latents1, latents2)
