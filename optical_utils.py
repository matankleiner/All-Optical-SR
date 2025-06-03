import torch
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


def NA_cutoff(data, circ):
    data_fft = torch.fft.fftshift(torch.fft.fft2(data), dim=(2,3))
    data_fft_filtered = data_fft * circ
    data_space_filtered = torch.fft.ifft2(torch.fft.ifftshift(data_fft_filtered, dim=(2,3)))
    return data_space_filtered
