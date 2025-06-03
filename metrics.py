import os 
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
from torchmetrics.image import StructuralSimilarityIndexMeasure

from model import Model
from utils import set_seed, QuickDrawData, EMNISTDataset
from optical_utils import circ, NA_cutoff, PIXEL_SIZE


def PSNR(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1 / mse)
    return psnr

def energy_compute(in_, out, pad, obj_shape):
    in_energy = torch.sum(in_[:, :, pad:pad+obj_shape, pad:pad+obj_shape]).mean()
    out_energy = torch.sum(out[:, :, pad:pad+obj_shape, pad:pad+obj_shape]).mean()
    energy = out_energy / in_energy
    return energy


def data_loader(dataset, batch_size, transforms_train):
    if dataset=='QuickDraw':
        _, loader = QuickDrawData(data_dir="/home/matan/Desktop/AONN/OpticalSR/data/QuickDraw",
                            max_examples_per_class=2000, batch_size=100, train_val_split_pct=.5,
                            transforms_train=transforms_train)
        return loader 
    else:
        if dataset=='MNIST':
            set = datasets.MNIST(root='../data', train=False, download=True, transform=transforms_train)
        elif dataset=='FashionMNIST':
            set = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transforms_train)
        elif dataset=='KMNIST':
            set = datasets.KMNIST(root='../data', train=False, download=True, transform=transforms_train)
        elif dataset=='EMNIST':
            set = EMNISTDataset('/home/matan/Desktop/AONN/OpticalSR/data/EMNIST/emnist-letters-test-images-idx3-ubyte.gz',
                    '/home/matan/Desktop/AONN/OpticalSR/data/EMNIST/emnist-letters-test-labels-idx1-ubyte.gz',
                    transform=transforms_train)
        
        loader = torch.utils.data.DataLoader(set, batch_size=batch_size, drop_last=True)

    
        return loader

if __name__ == "__main__":
    
    gpu_number = 0
    device = f"cuda:{gpu_number}"

    td = "MNIST" # dataset used for evaluation
    p = "trained_model_name"

    path = os.path.join("/path/to/your/trained/models", p, f"checkpoints/ckpt1000.pth")
    state = torch.load(path, map_location=device)

    set_seed(seed=state['args'].seed)

    state['args'].gpu_number = gpu_number

    pad = state['args'].pad
    obj_shape = state['args'].obj_shape
    SHAPE = obj_shape + 2 * pad 

    state['args'].dataset = td

    if state['args'].dataset == 'QuickDraw' or state['args'].dataset == 'EMNIST':
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((obj_shape, obj_shape), TF.InterpolationMode.NEAREST),
                                            transforms.Pad(padding=(pad, pad, pad, pad))])
        loader = data_loader(dataset=state['args'].dataset, batch_size=100, transforms_train=transforms_train)
        
    else:
        transforms_train = transforms.Compose([transforms.Resize((obj_shape, obj_shape), TF.InterpolationMode.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Pad(padding=(pad, pad, pad, pad))])
        
        loader = data_loader(dataset=state['args'].dataset, batch_size=100, transforms_train=transforms_train)


    model = Model(state['args']).to(device)
    model.load_state_dict(state['model'])

    cut_off_circ = circ(obj_shape * PIXEL_SIZE, SHAPE,
                        state['args'].out_dist / state['args'].radius, state['args'].wavelength,
                        state['args'].out_dist, device)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    model.eval()
    running_psnr = 0.
    running_ssim = 0.
    running_energy = 0.
    with torch.no_grad():  
        for data, _ in tqdm(loader):
            data = data.to(device)
            data_i = data
            t_i_data = torch.sqrt(data_i)
            lr_data = NA_cutoff(t_i_data, cut_off_circ)
            output = model(lr_data)
            energy = energy_compute(torch.abs(lr_data)**2, output, state['args'].pad, state['args'].obj_shape) 
            output = output / output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            psnr_value = PSNR(output[:,:,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape],
                            data[:,:,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape])
            ssim_value = ssim(output[:,:,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape],
                            data[:,:,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape])
            running_psnr += psnr_value
            running_ssim += ssim_value
            running_energy += energy
            

        # Normalizing the loss by the total number of train batches
        running_psnr /= len(loader)
        running_ssim /= len(loader)
        running_energy /= len(loader)

        print(f"Model {p} tested on: {td}. PSNR: {running_psnr:.4f}, SSIM: {running_ssim:.4f}, energy: {running_energy:.6f}")

