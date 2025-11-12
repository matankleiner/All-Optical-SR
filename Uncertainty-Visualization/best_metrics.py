import os 

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
import numpy as np
from tqdm import tqdm

from model import Model
from utils import set_seed, rs_tf_kernel, FreeSpaceProp, RandomSquareMasknFS
from optical_utils import circ, NA_cutoff, PIXEL_SIZE


def PSNR(pred, target):
    mse = F.mse_loss(pred, target)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


def data_loader(dataset, batch_size, transforms_train):
        set = datasets.MNIST(root='../data', train=False, download=True, transform=transforms_train)
        loader = torch.utils.data.DataLoader(set, batch_size=batch_size, drop_last=True)
        return loader

if __name__ == "__main__":
    
    gpu_number = 0
    device = f"cuda:{gpu_number}"
        
    p = "path/to/trained/model"
    ckpt = 1000
    path = os.path.join(p, f"checkpoints/ckpt{ckpt}.pth")
    state = torch.load(path, map_location=device)

    set_seed(seed=state['args'].seed)

    state['args'].gpu_number = gpu_number

    pad = state['args'].pad
    obj_shape = state['args'].obj_shape
    SHAPE = obj_shape + 2 * pad 
    
    transforms_train = transforms.Compose([transforms.Resize((obj_shape, obj_shape), TF.InterpolationMode.NEAREST),
                                        transforms.ToTensor(),
                                        transforms.Pad(padding=(pad, pad, pad, pad))])
    loader = data_loader(dataset=state['args'].dataset, batch_size=100, transforms_train=transforms_train)

    model = Model(state['args']).to(device)
    model.load_state_dict(state['model'])

    cut_off_circ = circ(obj_shape * PIXEL_SIZE, SHAPE//2,
                        state['args'].out_dist / state['args'].radius, state['args'].wavelength,
                        state['args'].out_dist, device)
    
    kernel = rs_tf_kernel(SHAPE, device)
    free_space_prop = FreeSpaceProp(kernel)
    
    model.eval()
    running_psnr = 0.
    best_head1 = 0
    best_head2 = 0
    best_head3 = 0
    best_head4 = 0
    with torch.no_grad():  
        for data, _ in tqdm(loader):
            data = data.to(device)
            data_i = data
            t_i_data = torch.sqrt(data_i)
            if state['args'].deg == 'lr':            
                deg_data = NA_cutoff(t_i_data, cut_off_circ, SHAPE)
            elif state['args'].deg == 'r_mask_fs':
                deg_data, _ = RandomSquareMasknFS(state['args'].random_mask_size, state['args'].pad, t_i_data.shape, device, free_space_prop, t_i_data)
            output = model(deg_data)
            output_norm = output.squeeze(0) / output.squeeze(0).max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            output1 = output_norm[:,:,state['args'].out_pad:state['args'].out_pad+state['args'].obj_shape,state['args'].out_pad:state['args'].out_pad+state['args'].obj_shape]
            output2 = output_norm[:,:,state['args'].out_pad:state['args'].out_pad+state['args'].obj_shape,state['args'].out_pad+state['args'].obj_shape:state['args'].out_pad+state['args'].obj_shape*2]
            output3 = output_norm[:,:,state['args'].out_pad+state['args'].obj_shape:state['args'].out_pad+state['args'].obj_shape*2,state['args'].out_pad:state['args'].out_pad+state['args'].obj_shape]
            output4 = output_norm[:,:,state['args'].out_pad+state['args'].obj_shape:state['args'].out_pad+state['args'].obj_shape*2,state['args'].out_pad+state['args'].obj_shape:state['args'].out_pad+state['args'].obj_shape*2]
            
            l1_distances1 = [torch.nn.functional.l1_loss(output1[i, 0, :, :], data[i, 0, pad:pad+obj_shape, pad:pad+obj_shape]).item() for i in range(data_i.size(0))]
            l1_distances2 = [torch.nn.functional.l1_loss(output2[i, 0, :, :], data[i, 0, pad:pad+obj_shape, pad:pad+obj_shape]).item() for i in range(data_i.size(0))]
            l1_distances3 = [torch.nn.functional.l1_loss(output3[i, 0, :, :], data[i, 0, pad:pad+obj_shape, pad:pad+obj_shape]).item() for i in range(data_i.size(0))]
            l1_distances4 = [torch.nn.functional.l1_loss(output4[i, 0, :, :], data[i, 0, pad:pad+obj_shape, pad:pad+obj_shape]).item() for i in range(data_i.size(0))]
        
            argmin_l1_distances = [np.argmin([l1_distances1[i], l1_distances2[i], l1_distances3[i], l1_distances4[i]]) for i in range(len(l1_distances1))]
            for j, argmin_ in enumerate(argmin_l1_distances):
                psnr = 0
                if argmin_ == 0:
                    psnr = PSNR(output1[j,0,:,:],
                            data[j,0,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape])
                    best_head1 += 1
                elif argmin_ == 1:
                    psnr = PSNR(output2[j,0,:,:],
                            data[j,0,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape])
                    best_head2 += 1
                elif argmin_ == 2:
                    psnr = PSNR(output3[j,0,:,:],
                            data[j,0,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape])
                    best_head3 += 1
                elif argmin_ == 3:
                    psnr = PSNR(output4[j,0,:,:],
                            data[j,0,state['args'].pad:state['args'].pad+state['args'].obj_shape,state['args'].pad:state['args'].pad+state['args'].obj_shape])
                    best_head4 += 1
                running_psnr += psnr
            running_psnr /= len(loader)

        print(f"For {p}, PSNR Best: {running_psnr:.4f}, best 1 {best_head1}, best 2 {best_head2}, best 3 {best_head3}, best 4 {best_head4}")

