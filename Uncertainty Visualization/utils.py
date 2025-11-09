import random
import time
import os
import torch
import numpy as np

from optical_utils import NA_cutoff, FreeSpaceProp, rs_tf_kernel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data(dataset, batch_size, split, transforms_train):
    if split == 'train':
        set = eval(f"datasets.{dataset}(root='../data', train=True, download=True, transform=transforms_train)")
        samples = torch.randperm(set.data.shape[0])[:10000]
    elif split == 'test':
        set = eval(f"datasets.{dataset}(root='../data', train=False, download=True, transform=transforms_train)")
        samples = torch.randperm(set.data.shape[0])[:1000]
    loader = torch.utils.data.DataLoader(set, batch_size=batch_size, drop_last=True, sampler=samples)
    return loader
   

def oracle_loss(output, target, criterion, epoch, args):
    
    output_norm = output.squeeze(0) / output.squeeze(0).max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
    
    output1 = output_norm[:,:,args.out_pad:args.out_pad+args.obj_shape,args.out_pad:args.out_pad+args.obj_shape]
    output2 = output_norm[:,:,args.out_pad:args.out_pad+args.obj_shape,args.out_pad+args.obj_shape:args.out_pad+args.obj_shape*2]
    output3 = output_norm[:,:,args.out_pad+args.obj_shape:args.out_pad+args.obj_shape*2,args.out_pad:args.out_pad+args.obj_shape]
    output4 = output_norm[:,:,args.out_pad+args.obj_shape:args.out_pad+args.obj_shape*2,args.out_pad+args.obj_shape:args.out_pad+args.obj_shape*2]

    target_ = target[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape]

    loss1 = criterion(output1, target_)
    loss2 = criterion(output2, target_)
    loss3 = criterion(output3, target_)
    loss4 = criterion(output4, target_)
    
    
    if epoch < args.epcoh_initialize:
        loss = (loss1.mean() + loss2.mean() + loss3.mean() + loss4.mean()) / 4
        return loss
    else: 
        losses = torch.stack([loss1, loss2, loss3, loss4])
        losses = losses.mean(dim=[2, 3, 4])
        min_losses, _ = torch.min(losses, dim=0)  

        return min_losses.mean()


def RandomSquareMasknFS(mask_size, pad, shape, device, free_space_prop, x):
    mask = torch.ones(shape, device=device)
    _, _, h, w = shape
    h = h - 2 * pad
    w = w - 2 * pad
    top_left_x = random.randint(0, w - mask_size)
    top_left_y = random.randint(0, h - mask_size)
    mask[:, :, pad + top_left_y:pad + top_left_y + mask_size, pad + top_left_x:pad + top_left_x + mask_size] = 0 
    x = free_space_prop(x)
    x = x * mask
    x = free_space_prop(x)
    return x, mask


def training(
        model,
        cut_off_circ,
        args,
        shape,
        loader,
        training_criterion,
        optimizer,
        scheduler,
        device,
):

    # training loop
    if args.deg == 'r_mask_fs':
        kernel = rs_tf_kernel(shape, device)
        free_space_prop = FreeSpaceProp(kernel)
    
    for epoch in range(1, args.epochs + 1):
        model.train() 
        running_loss = 0.0
        epoch_time = time.time()
        for t_data, _ in loader:
            t_data = t_data.to(device)
            for i in range(4):               
                t_i_data = t_data[i*10:10*(1+i),:,:,:]
                t_i_data_sqrt = torch.sqrt(t_i_data)
                if args.deg == 'lr':            
                    deg_data = NA_cutoff(t_i_data_sqrt, cut_off_circ, shape)
                elif args.deg == 'r_mask_fs':
                    deg_data, _ = RandomSquareMasknFS(args.random_mask_size, args.pad, t_i_data_sqrt.shape, device, free_space_prop, t_i_data_sqrt)
                deg_data = deg_data.to(device)
                output = model(deg_data)
                loss = oracle_loss(output, t_i_data, training_criterion, epoch, args)
                               
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.data.item()
        scheduler.step()
        # Normalizing the loss by the total number of train batches
        running_loss /= len(loader)

        epoch_time = time.time() - epoch_time
        print(f"Epoch: {epoch:0>2}/{args.epochs} | Training Loss: {running_loss:.4f} | Epoch Time: {epoch_time:.2f} secs")

        if epoch % 100 == 0 or epoch == args.epochs:
            print('==> Saving model ...')
            state = {'model': model.state_dict(), 'epoch': epoch, 'args' : args}
            os.makedirs(f'trials/{args.trial_name}/checkpoints', exist_ok=True)
            torch.save(state, f'./trials/{args.trial_name}/checkpoints/ckpt{epoch}.pth')
