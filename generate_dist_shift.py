import torch
import numpy as np
import pickle
from torchvision.utils import make_grid
from sampling import edm_sample
from matplotlib import pyplot as plt
from pathlib import Path
import click
import time

import sys
import os
sys.path.insert(1, os.getcwd() + '/edm')
import dnnlib


def trunc_coords(shape: tuple, trunc: int):
    """
    Samples latent codes for the truncation data.
    Args:
        shape: a tuple depicting the shape of the latent codes, usually a tuple of size [N, 3, h, w]
        trunc: the amount of truncation to apply - the controlling parameter for the distribution shift

    Returns:
        the generated latent codes, a torch Tensor with the same shape as the "shape" input

    """
    return torch.randn(shape)*(trunc/100)


def extend_coords(shape: tuple, value: int, target=None):
    """
    Samples latent codes for the extend data. For this data, a "target point" is defined, after a spherical
    interpolation is applied between this target and random samples from a standard normal distribution. The amount of
    interpolation is defined by the "value" argument.
    Args:
        shape: a tuple depicting the shape of the latent codes, usually a tuple of size [N, 3, h, w]
        value: the controlling parameter for the distribution shift, an int between 50 and 100
        target: the target for the spherical interpolation; if no target is supplied, one is sampled with a fixed seed

    Returns:
        the generated latent codes, a torch Tensor with the same shape as the "shape" input

    """
    t = value/100
    zs = torch.randn(shape)
    if target is None:
        state = np.random.get_state()
        np.random.seed(45678)
        target = torch.from_numpy(np.random.randn(*shape[1:])).float()[None]
        np.random.set_state(state)

    dot = torch.sum(zs*target, dim=1)/(torch.norm(zs, dim=1)*torch.norm(target, dim=1))
    omega = torch.acos(dot)[:, None]
    so = torch.sin(omega)
    return torch.sin((1-t)*omega)*target/so + torch.sin(t*omega)*zs/so


def overlap_coords(shape: tuple, value: int):
    """
    Samples latent codes for the overlap data. For this data, a spherical interpolation between two targets is defined
    according to the "value" parameters, after which a spherical interpolation of 50% between this target and samples
    from a standard normal are returned.
    Args:
        shape: a tuple depicting the shape of the latent codes, usually a tuple of size [N, 3, h, w]
        value: the controlling parameter for the distribution shift, an int between 0 and 100

    Returns:
        the generated latent codes, a torch Tensor with the same shape as the "shape" input

    """
    state = np.random.get_state()
    np.random.seed(45678)
    target1 = torch.from_numpy(np.random.randn(*shape[1:])).float()
    target2 = torch.from_numpy(np.random.randn(*shape[1:])).float()
    np.random.set_state(state)
    
    t = value/100
    dot = torch.sum(target1 * target2) / (torch.norm(target1) * torch.norm(target2))
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    target = torch.sin((1-t)*omega)*target1/so + torch.sin(t*omega)*target2/so
    return extend_coords(shape, value=50, target=target[None])


def generate_cifar(z_func, N: int, bs: int, its: int, save_path: str,
                      network_pkl: str='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/'):
    """
    Generate images from EDM's CIFAR10 model
    Args:
        z_func: function for sampling from the latent space should be used (defined according to the experiment type)
        N: number of samples to generate
        bs: the batch size of images to generate at any point
        its: number of iterations to run the diffusion model
        save_path: saving path for generated images
        network_pkl: path to pretrained EDM network

    Returns:
        the images, as a numpy array with shape [N, 32, 32, 3], and labels, as a numpy array with shape [N,]

    """
    with dnnlib.util.open_url(network_pkl + 'edm-cifar10-32x32-cond-vp.pkl', verbose=False) as f:
        net = pickle.load(f)['ema'].to('cuda')

    full_labels = np.random.choice(10, N).astype(int)
    images = None
    start_time = time.time()
    for i in range(0, N, bs):
        elapsed = time.time() - start_time
        if i > 0: towait = (N-i)*elapsed/i
        else: towait = 0
        print(f'{i}/{N}, '
              f'elapsed: {int(elapsed//(60*60)):d}h '
                       f'{int(elapsed//60)%60:02d}m '
                       f'{int(elapsed)%60:02d}s, '
              f'left: {int(towait//(60*60)):d}h '
                    f'{int(towait//60)%60:02d}m '
                    f'{int(towait)%60:02d}s',
              flush=True)
        labs = torch.zeros(bs, 10, device='cuda')
        for j in range(bs): labs[j, full_labels[i+j]] = 1
        lats = z_func(bs)

        ims = edm_sample(net, lats, class_labels=labs, num_steps=its, verbose=False)[0]*.5 + .5
        ims = ims.clamp(0, 1)*255
        ims = ims.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        if images is not None: images = np.concatenate([images, ims], axis=0)
        else: images = ims

        with open(save_path, 'wb') as f: pickle.dump((images, full_labels[:len(images)]), f)
    return images, full_labels


def generate_imagenet(z_func, N: int, bs: int, its: int, save_path: str,
                      network_pkl: str='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/'):
    """
    Generate images from EDM's ImageNet model
    Args:
        z_func: function for sampling from the latent space should be used (defined according to the experiment type)
        N: number of samples to generate
        bs: the batch size of images to generate at any point
        its: number of iterations to run the diffusion model
        save_path: saving path for generated images
        network_pkl: path to pretrained EDM network

    Returns: the images, as a numpy array with shape [N, 64, 64, 3], and labels, as a numpy array with shape [N,]

    """
    with dnnlib.util.open_url(network_pkl + 'edm-imagenet-64x64-cond-adm.pkl', verbose=False) as f:
        net = pickle.load(f)['ema'].to('cuda')

    full_labels = np.random.choice(10, N).astype(int)

    images = None
    start_time = time.time()
    for i in range(0, N, bs):
        elapsed = time.time() - start_time
        if i > 0: towait = (N-i)*elapsed/i
        else: towait = 0
        print(f'{i}/{N}, '
              f'elapsed: {int(elapsed//(60*60)):d}h '
                       f'{int(elapsed//60)%60:02d}m '
                       f'{int(elapsed)%60:02d}s, '
              f'left: {int(towait//(60*60)):d}h '
                    f'{int(towait//60)%60:02d}m '
                    f'{int(towait)%60:02d}s',
              flush=True)
        labs = torch.zeros(bs, net.label_dim, device='cuda')
        for j in range(bs): labs[j, full_labels[i+j]*100 + 1] = 1
        lats = z_func(bs)

        ims = edm_sample(net, lats, class_labels=labs, num_steps=its, verbose=False)[0]*.5 + .5
        ims = ims.clamp(0, 1)*255
        ims = ims.cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        if images is not None: images = np.concatenate([images, ims], axis=0)
        else: images = ims

        with open(save_path, 'wb') as f: pickle.dump((images, full_labels[:len(images)]), f)
    return images, full_labels


@click.command()
@click.option('--dataset',  help='dataset to generate', type=str, required=True)
@click.option('--exp',      help='experiment to generate, one of ["trunc", "overlap", "extend"]', type=str, required=True)
@click.option('--value',    help='number controlling the amount of shift in the experiment', type=int, required=True)
@click.option('--n_steps',  help='number of diffusion iterations', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--n_train',  help='number of training examples to generate', type=int, default=50000, show_default=True)
@click.option('--n_val',    help='number of validation examples to generate', type=int, default=1000, show_default=True)
@click.option('--n_test',   help='number of test examples to generate', type=int, default=5000, show_default=True)
@click.option('--batch',    help='batch size for image generation', type=int, default=100, show_default=True)
@click.option('--root',     help='root path for saving the generated data', type=str, default='data/', show_default=True)
@click.option('--name',     help='folder name', type=str, default=None, show_default=True)
def main(dataset: str, exp: str, value: int, **kwargs):
    print(f'dataset={dataset}, exp={exp}, value={value}')
    print(kwargs, flush=True)

    dataset, exp = dataset.lower(), exp.lower()
    name = kwargs['name'] if kwargs['name'] is not None else exp
    save_path = kwargs['root'] + f'{dataset}/{name}/{str(value).replace(".", "_")}/'

    shape = [3, 32, 32] if dataset=='cifar10' else [3, 64, 64]
    dim = np.prod(shape)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    if exp == 'trunc':
        z_func = lambda k: trunc_coords((k, dim), trunc=value).reshape(k, *shape).to('cuda')
    elif exp == 'extend':
        z_func = lambda k: extend_coords((k, dim), value=value).reshape(k, *shape).to('cuda')
    elif exp == 'overlap':
        z_func = lambda k: overlap_coords((k, dim), value=value).reshape(k, *shape).to('cuda')
    else:
        raise NotImplementedError

    # generate training data
    if (exp == 'trunc' and value == 90) \
            or (exp == 'overlap' and value == 0) \
            or (exp == 'extend' and value == 50) \
            and kwargs['n_train'] > 0:

        print('Generating training set', flush=True)
        torch.manual_seed(12345)
        np.random.seed(12345)
        if dataset == 'cifar10': generate_cifar(z_func, N=kwargs['n_train'], bs=kwargs['batch'],
                                                its=kwargs['n_steps'], save_path=save_path + 'train.pkl')
        elif dataset == 'imagenet': generate_imagenet(z_func, N=kwargs['n_train'], bs=kwargs['batch'],
                                                      its=kwargs['n_steps'], save_path=save_path + 'train.pkl')
        else: raise NotImplementedError

        print('Generating validation set', flush=True)
        torch.manual_seed(54321)
        np.random.seed(54321)
        if dataset == 'cifar10': generate_cifar(z_func, N=kwargs['n_val'], bs=kwargs['batch'],
                                                its=kwargs['n_steps'], save_path=save_path + 'val.pkl')
        elif dataset == 'imagenet': generate_imagenet(z_func, N=kwargs['n_val'], bs=kwargs['batch'],
                                                      its=kwargs['n_steps'], save_path=save_path + 'val.pkl')
        else: raise NotImplementedError

    torch.manual_seed(0)
    np.random.seed(0)
    print('Generating test set', flush=True)
    if dataset == 'cifar10':
        test_ims, test_labs = generate_cifar(z_func, N=kwargs['n_test'], bs=kwargs['batch'],
                                             its=kwargs['n_steps'], save_path=save_path+'test.pkl')
    elif dataset == 'imagenet':
        test_ims, test_labs = generate_imagenet(z_func, N=kwargs['n_test'], bs=kwargs['batch'],
                                                its=kwargs['n_steps'], save_path=save_path+'test.pkl')
    else: raise NotImplementedError

    sample = make_grid(torch.from_numpy(test_ims[:49]/255.).float().permute(0, -1, 1, 2), nrow=7,
                       pad_value=1).permute(1, 2, 0).numpy()
    plt.imsave(save_path+'sample.png', sample)


if __name__ == "__main__":
    main()
