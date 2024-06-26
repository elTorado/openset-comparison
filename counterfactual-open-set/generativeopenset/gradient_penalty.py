import torch
from torch import autograd
from vast.tools import set_device_gpu, set_device_cpu, device

def calc_gradient_penalty(netD, real_data, fake_data, args, penalty_lambda=10.0):
        if args['gpu'] is not None:
                set_device_gpu(index=args['gpu'] )
        else:
                set_device_cpu()        

                
        alpha = torch.rand(real_data.size()[0], 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        lpha = device(alpha)
        
        # Traditional WGAN-GP
        interpolates = alpha * real_data + (1 - alpha) * fake_data

        interpolates = device(interpolates)
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        ones = torch.ones(disc_interpolates.size())
        ones = device(ones)
        
        gradients = autograd.grad(
                outputs=disc_interpolates, 
                inputs=interpolates, 
                grad_outputs=ones, 
                create_graph=True, 
                retain_graph=True, 
                only_inputs=True)[0]

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda
        return penalty
