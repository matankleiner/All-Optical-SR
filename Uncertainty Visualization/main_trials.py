import os
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import Model
from utils import set_seed, data, training
from optical_utils import circ
from config import get_args, PIXEL_SIZE


if __name__ == '__main__':
    parsed = get_args()
    set_seed(seed=parsed.seed)

    # load available device
    device = torch.device(f"cuda:{parsed.gpu_number}" if torch.cuda.is_available() else "cpu")
    print("current device:", device)

    SHAPE = parsed.obj_shape + 2 * parsed.pad 
    
    transforms_train = transforms.Compose([transforms.Resize((parsed.obj_shape, parsed.obj_shape), TF.InterpolationMode.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Pad(padding=(parsed.pad, parsed.pad, parsed.pad, parsed.pad))])

    train_loader = data(dataset=parsed.dataset, batch_size=parsed.batch_size, split='train', transforms_train=transforms_train)
    test_loader = data(dataset=parsed.dataset, batch_size=parsed.batch_size, split='test', transforms_train=transforms_train)
    
    model = Model(parsed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parsed.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if parsed.continue_training:
        checkpoint = torch.load(parsed.ckpt_path)
        model.load_state_dict(checkpoint[f'models'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        parsed.epoch = checkpoint['epoch']
        
    print(model.parameters)

    test_criterion = nn.MSELoss(reduction='none')
    if parsed.training_criterion == 'l2':
        training_criterion = nn.MSELoss(reduction='none')
    elif parsed.training_criterion == 'l1':
        training_criterion = nn.L1Loss(reduction='none')
    cut_off_circ = circ(parsed.obj_shape * PIXEL_SIZE, SHAPE//2, parsed.out_dist / parsed.radius, parsed.wavelength, parsed.out_dist, device)
    training(model, cut_off_circ, parsed, SHAPE, train_loader, training_criterion, optimizer, scheduler, device)
    if not os.path.isdir(f'trials/{parsed.trial_name}'):
        os.mkdir(f'trials/{parsed.trial_name}')
