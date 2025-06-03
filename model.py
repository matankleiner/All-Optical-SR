import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLinearLayer(nn.Module):
    def __init__(self, args):
        super(NonLinearLayer, self).__init__()
        self.phase_filter = torch.nn.Parameter(torch.randn(size=(1, args.phase_filter_shape, args.phase_filter_shape)))
        self.pad = args.pad
        self.w = args.nonlinear_weight
        self.rect = F.pad(torch.ones(1, 1, args.obj_shape, args.obj_shape),
                          pad=(args.pad, args.pad, args.pad, args.pad)).to(f'cuda:{args.gpu_number}')

    def forward(self, x):
        x1 = torch.fft.fftshift(torch.fft.fft2(x), dim=(2,3))
        x2 = x1 * F.pad(torch.exp(1j * self.phase_filter), pad=(self.pad, self.pad, self.pad, self.pad))
        x3 = torch.fft.ifft2(torch.fft.ifftshift(x2, dim=(2,3)))
        x4 = x3 * self.rect
        intensity = torch.abs(x4)**2
        x5 = x4 * torch.exp(1j * self.w * intensity)    
        return x5
    

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.model = self._create_model()

    def _create_model(self):       
        layers = []
        for _ in range(self.args.num_layers):
            layers.append(NonLinearLayer(self.args))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.model(x)
        return torch.abs(out)**2
    