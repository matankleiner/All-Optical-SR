import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np

import gzip
import random
import time
import os

from optical_utils import NA_cutoff
from QuickDrawDataset import QuickDrawDataset


class EMNISTDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        with gzip.open(img_path, 'r') as imgfile:
            self.images = np.frombuffer(imgfile.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

        with gzip.open(label_path, 'r') as labelfile:
            self.labels = np.frombuffer(labelfile.read(), dtype=np.uint8, offset=8)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

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


def QuickDrawData(data_dir, max_examples_per_class, batch_size, train_val_split_pct, transforms_train):
    ds = QuickDrawDataset(data_dir, max_examples_per_class, transforms=transforms_train)
    train_ds, test_ds = ds.split(train_val_split_pct)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, drop_last=True)
    return train_loader, test_loader


def training(
        model,
        cut_off_circ,
        args,
        loader,
        test_loader,
        training_criterion,
        test_criterion,
        optimizer,
        scheduler,
        device,
):

    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train() 
        running_loss = 0.0
        epoch_time = time.time()
        for t_data, _ in loader:
            t_data = t_data.to(device)
            # gradient accumlatio
            for i in range(4):               
                t_i_data = t_data[i*10:10*(1+i),:,:,:]
                t_i_data_sqrt = torch.sqrt(t_i_data)
                lr_data = NA_cutoff(t_i_data_sqrt, cut_off_circ)
                output = model(lr_data)
                if args.gamma == 0:
                    output = output / output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                    loss = training_criterion(output[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape],
                                          t_i_data[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape])     
                else:
                    output_norm = output / output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                    distortion_loss = training_criterion(output_norm[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape],
                                          t_i_data[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape]) 
                    power_loss = torch.sum(output[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape])
                    loss = distortion_loss - args.gamma * power_loss
               
                # backward pass
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.data.item()
        scheduler.step()
        # normalizing the loss by the total number of train batches
        running_loss /= len(loader)

        # if epoch % 10 == 0 or epoch == args.epochs:
        epoch_time = time.time() - epoch_time
        print(f"Epoch: {epoch:0>2}/{args.epochs} | Training Loss: {running_loss:.4f} | Epoch Time: {epoch_time:.2f} secs")

        if epoch % 100 == 0 or epoch == args.epochs:
            print('==> Saving model ...')
            state = {'model': model.state_dict(), 'epoch': epoch, 'args' : args}
            os.makedirs(f'trials/{args.trial_name}/checkpoints', exist_ok=True)
            torch.save(state, f'./trials/{args.trial_name}/checkpoints/ckpt{epoch}.pth')
            test_loss = evaluation(model, cut_off_circ, test_loader, test_criterion, device)
            print(f"Test Loss: {test_loss}")
           


def evaluation(model,
               cut_off_circ,
               loader,  
               criterion,
               device,
):

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for t_data, _ in loader:
            t_data = t_data.to(device)
            t_i_data = torch.sqrt(t_data)
            lr_data = NA_cutoff(t_i_data, cut_off_circ)
            output = model(lr_data)
            output = output / output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            loss = criterion(output, t_data)
            running_loss += loss.data.item()

    # Normalizing the loss by the total number of train batches
    running_loss /= len(loader)

    return running_loss


def ir_training(
        model,
        cut_off_circ,
        args,
        loader,
        test_loader,
        training_criterion,
        test_criterion,
        optimizer,
        scheduler,
        device,
):

    # training loop
    blind_intensities = [4., 3., 2., 1., 0.75, 0.5, 0.25, 0.1]
    for epoch in range(1, args.epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        intensity = random.randint(0, len(blind_intensities)-1)
        for t_data, _ in loader:
            t_data = t_data.to(device)
            for i in range(4):               
                t_i_data = t_data[i*10:10*(1+i),:,:,:]
                t_i_data_intensity = t_i_data * blind_intensities[intensity]
                t_i_data_sqrt = torch.sqrt(t_i_data_intensity)
                lr_data = NA_cutoff(t_i_data_sqrt, cut_off_circ)
                output = model(lr_data)
                if args.gamma == 0:
                    output = (output / output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]) * blind_intensities[intensity]
                    loss = training_criterion(output[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape],
                                          t_i_data_intensity[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape]) * (1 / blind_intensities[intensity]) 
                else:
                    output_norm = (output / output.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]) * blind_intensities[intensity]
                    distortion_loss = training_criterion(output_norm[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape],
                                          t_i_data_intensity[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape]) * (1 / blind_intensities[intensity]) 
                    power_loss = torch.sum(output[:,:,args.pad:args.pad+args.obj_shape,args.pad:args.pad+args.obj_shape])
                    loss = distortion_loss - args.gamma * power_loss
                
                # backward pass
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.data.item()
        scheduler.step()
        # normalizing the loss by the total number of train batches
        running_loss /= len(loader)

        # if epoch % 10 == 0 or epoch == args.epochs:
        epoch_time = time.time() - epoch_time
        print(f"Epoch: {epoch:0>2}/{args.epochs} | Training Loss: {running_loss:.4f} | Epoch Time: {epoch_time:.2f} secs")

        if epoch % 100 == 0 or epoch == args.epochs:
            print('==> Saving model ...')
            state = {'model': model.state_dict(), 'epoch': epoch, 'args' : args}
            os.makedirs(f'trials/{args.trial_name}/checkpoints', exist_ok=True)
            torch.save(state, f'./trials/{args.trial_name}/checkpoints/ckpt{epoch}.pth')
            test_loss = evaluation(model, cut_off_circ, test_loader, test_criterion, device)
            print(f"Test Loss: {test_loss}")
