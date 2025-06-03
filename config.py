import argparse

PIXEL_SIZE = 4*10e-7

def get_args():
    parser = argparse.ArgumentParser()
    
    # hyperparameters 
    parser.add_argument('--batch_size', help='batch size', default=40, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
    parser.add_argument('--epochs', help='number of epochs', default=1000, type=int)
    
    # NA and wavelength 
    parser.add_argument('--out_dist', help='distance between optical system output and output plane, in meter', default=1e-2, type=float)
    parser.add_argument('--wavelength', help='wavelength to use', default=550e-09, type=float)    
    parser.add_argument('--radius', help='exit pupil radius', default=60, type=int)    
    
    # trial properties
    parser.add_argument('--obj_shape', help='object shape', default=112, type=int)
    parser.add_argument('--pad', help='padding', default=112, type=int)
    parser.add_argument('--phase_filter_shape', help='phase filter shape', default=112, type=int)
    parser.add_argument('--num_layers', help='number of blocks', default=10, type=int)
    parser.add_argument('--gamma', help='weight for power preservation regularization term', default=1e-06, type=float)
    parser.add_argument('--ir_training', help='intensity robust training', action='store_true')   
    # activation function 
    parser.add_argument('--nonlinear_weight', help='', default=4., type=float)
    
    # dataset to use 
    parser.add_argument('--dataset', help='dataset to use, between MNIST, FashionMNIST and Quick, Draw!', default='MNIST', type=str)
    
    # trial configurations      
    parser.add_argument('--seed', help='', default=42, type=int)
    parser.add_argument('--trial_name', help='the trial name', default='sr_trial', type=str)
    parser.add_argument('--gpu_number', help='gpu to use', default=0, type=int) 
   
    # continue training 
    parser.add_argument('--continue_training', help='continue training from a checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', help='path to checkpoint', type=str)
    
    return parser.parse_args()
