import torch
from tqdm import tqdm


def edm_schedule(num_steps: int, sigma_min: float=.002, sigma_max: float=80, rho: float=7,
                 device='cuda'):
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1])*1e-5])  # t_N = 0
    return t_steps


def edm_sample(
        net, latents, class_labels=None, num_steps=18,
        sigma_min: float=0.002, sigma_max: float=80, rho: float=7,
        verbose: bool=True, invert: bool=False, euler: bool=False,
):
    """
    An adaptation of the code used for sampling in the original EDM code.
    Adapted from: https://github.com/NVlabs/edm/blob/main/generate.py
    Args:
        net: the score network to use for sampling
        latents: initial latent coordinates from which sampling should start, a tensor with shape [N, d1, d2, ...]
        class_labels: the labels to generate, a tensor class numbers of length N
        num_steps: number of steps to use in the sampling procedure
        sigma_min: standard deviation of noise at the end of sampling
        sigma_max: standard deviation of noise assumed at the start of sampling
        rho: controls the scheduling of the sampler
        verbose: a boolean indicating whether a progress bar should displayed while sampling
        invert: a boolean indicating whether the sampling process should be inverted (image -> noise)
        euler: a boolean indicating whether to use the 1st order Euler algorithm instead of the one suggested in EDM

    Returns: a tuple with
                - the sampled images, as a tensor with shape [N, d1, d2, ...]
                - a list of length num_steps of the progress during sampling (the i-th element are the images at step i)

    """
    if invert and latents.min() > 0: latents = 2*latents - 1

    # Adjust noise levels based on what's supported by the network.
    t_steps = edm_schedule(num_steps=num_steps,
                           sigma_min=max(sigma_min, net.sigma_min),
                           sigma_max=min(sigma_max, net.sigma_max),
                           rho=rho)

    if invert: t_steps = t_steps.flip(dims=[0])

    # Main sampling loop.
    if not invert: x_next = latents.float() * t_steps[0]
    else: x_next = latents.float()
    steps = []

    pbar = tqdm(zip(t_steps[:-1], t_steps[1:]), disable=not verbose, total=len(t_steps[1:]))
    for i, (t_cur, t_next) in enumerate(pbar): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        t_hat = net.round_sigma(t_cur)
        pbar.set_postfix_str(f't = {t_hat:.3f}')

        # Euler step.
        denoised = net(x_cur, t_hat, class_labels).float()
        d_cur = (x_cur - denoised) / t_hat
        x_next = x_cur + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1 and not euler:
            denoised = net(x_next, t_next, class_labels).float()
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        steps.append(x_next.clone().detach().cpu())
    return x_next, steps

