import argparse

PIXEL_SIZE = 4*10e-7

def get_args():
    parser = argparse.ArgumentParser()
    
    # hyperparameters 
    parser.add_argument('--batch_size', help='batch size', default=40, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
    parser.add_argument('--epochs', help='number of epochs', default=1000, type=int)
    parser.add_argument('--epcoh_initialize', help='use regular loss', default=500, type=int)
    
    
    # trial properties
    parser.add_argument('--obj_shape', help='object shape', default=112, type=int)
    parser.add_argument('--out_pad', help='padding', default=224, type=int)
    parser.add_argument('--pad', help='padding', default=280, type=int)
    parser.add_argument('--phase_filter_shape', help='phase filter shape', default=224, type=int)
    parser.add_argument('--num_layers', help='number of enocder blocks', default=10, type=int)
    parser.add_argument('--training_criterion', help='training criterion to use', default='l1', type=str)
    parser.add_argument('--deg', help='[lr, r_mask_fs]', default='lr', type=str)
    parser.add_argument('--random_mask_size', help='mask size', default=80, type=int)
    parser.add_argument('--out_dist', help='distance between optical system output and output plane, in meter', default=1e-2, type=float)
    parser.add_argument('--wavelength', help='wavelength to use', default=550e-09, type=float)    
    parser.add_argument('--radius', help='exit pupil radius', default=180, type=int)    
    # activation function 
    parser.add_argument('--nonlinear_weight', help='', default=1., type=float)
    
    # dataset to use 
    parser.add_argument('--dataset', help='dataset to use', default='MNIST', type=str)
    
    # trial configurations      
    parser.add_argument('--seed', help='', default=42, type=int)
    parser.add_argument('--trial_name', help='the trial name', default='sr_trial', type=str)
    parser.add_argument('--gpu_number', help='gpu to use', default=0, type=int) 
   
    # continue training 
    parser.add_argument('--continue_training', help='continue training from a checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', help='path to checkpoint', type=str)
    
    return parser.parse_args()
