import os
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import Model
from utils import set_seed, data, QuickDrawData, training, ir_training
from optical_utils import circ
from config import get_args, PIXEL_SIZE


if __name__ == '__main__':
    parsed = get_args()
    set_seed(seed=parsed.seed)

    # load available device
    device = torch.device(f"cuda:{parsed.gpu_number}" if torch.cuda.is_available() else "cpu")
    print("current device:", device)

    SHAPE = parsed.obj_shape + 2 * parsed.pad 
    
    # load data
    if parsed.dataset == 'QuickDraw':
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((parsed.obj_shape, parsed.obj_shape), TF.InterpolationMode.NEAREST),
                                           transforms.Pad(padding=(parsed.pad, parsed.pad, parsed.pad, parsed.pad))])

        train_loader, test_loader = QuickDrawData(data_dir="/path/to/your/data/QuickDraw",
                                     max_examples_per_class=1100, batch_size=parsed.batch_size, train_val_split_pct=.1,
                                     transforms_train=transforms_train)  

    else:
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
        
    test_criterion = nn.MSELoss()
    training_criterion = nn.L1Loss()
    cut_off_circ = circ(parsed.obj_shape * PIXEL_SIZE, SHAPE, parsed.out_dist / parsed.radius, parsed.wavelength, parsed.out_dist, device)

    if parsed.ir_training:
        ir_training(model, cut_off_circ, parsed, train_loader, test_loader, training_criterion, test_criterion, optimizer, scheduler, device)
    else:
        training(model, cut_off_circ, parsed, train_loader, test_loader, training_criterion, test_criterion, optimizer, scheduler, device)
    
    if not os.path.isdir(f'trials/{parsed.trial_name}'):
        os.mkdir(f'trials/{parsed.trial_name}')
