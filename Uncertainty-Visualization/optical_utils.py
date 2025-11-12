import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import PIXEL_SIZE


def circ(size, shape, radius, wavelength, output_distance, device):
    fx = torch.arange((-1 / (2 * (PIXEL_SIZE))), (1 / (2 * (PIXEL_SIZE))), (1/size), device=device)
    Fx, Fy = torch.meshgrid(fx, fx, indexing="ij")
    f0 = radius / (wavelength * output_distance)
    arg = torch.sqrt(Fx ** 2 + Fy ** 2) / f0
    cut_off_circ = torch.where(arg < 1, 1, 0)    
    pad = int((shape - cut_off_circ.shape[0]) / 2) 
    cut_off_circ_pad = torch.nn.functional.pad(cut_off_circ, pad=(pad, pad, pad, pad)) 
    return cut_off_circ_pad


def NA_cutoff(data, circ, shape):
    data_fft = torch.fft.fftshift(torch.fft.fft2(data[:,:,shape//4:-shape//4,shape//4:-shape//4]), dim=(2,3))
    data_fft_filtered = data_fft * circ
    data_space_filtered = F.pad(torch.fft.ifft2(torch.fft.ifftshift(data_fft_filtered, dim=(2,3))), pad=(shape//4, shape//4, shape//4, shape//4))
    return data_space_filtered


def rs_tf_kernel(shape, device, wavelength=550e-09, dist=0.005):
    size = shape * PIXEL_SIZE
    fx = torch.arange((-1 / (2 * (PIXEL_SIZE))), (1 / (2 * (PIXEL_SIZE))), (1/(size)))
    Fx, Fy = torch.meshgrid(fx, fx, indexing="ij")
    k = 2 * np.pi / wavelength
    H = torch.exp(1j * k * dist * torch.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))
    H = H * torch.where(Fx**2 + Fy**2 < (1/wavelength)**2, 1, 0)
    return torch.fft.fftshift(H).to(device)


class FreeSpaceProp(nn.Module):
    def __init__(self, kernel_fft):
        super().__init__()
        self.kernel_fft = kernel_fft

    def forward(self, x):
        x_fft = torch.fft.fft2(torch.fft.fftshift(x, dim=(2,3)))
        out = torch.fft.ifftshift(torch.fft.ifft2(x_fft * self.kernel_fft), dim=(2,3))
        return out
    