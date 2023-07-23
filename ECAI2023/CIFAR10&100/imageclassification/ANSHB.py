import torch
import math
import numpy as np
import copy
from torch.optim.optimizer import Optimizer, required

class NHBAdp(Optimizer):
    """
    Normalized Stochastic Heavy ball with adaptive momentum
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.)
        delta: delta in NHBAdp 
    Example:
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backwad()
        >>> optimizer.step()
    Formula:
        v_{t+1} = (momentum)*beta_k*v_t + (1-beta_k)*g_t
        p_{t+1} = p_t - lr*v_{t+1}
        beta_k=Proj_[0,1-eps]((1-alpha*(grad_diff/x_diff))/(1+alpha*(grad_diff/x_diff)))
        where
        grad_diff = np.sqrt(np.dot(grad - grad_prev, grad - grad_prev))
        x_diff = np.sqrt(np.dot(x - x_prev, x - x_prev))
    """
    def __init__(self, params, lr=required, weight_decay=5e-4, delta=1e-3): 
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, weight_decay=weight_decay, delta=delta)
        super(NHBAdp, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(NHBAdp, self).__setstate__(state)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if group['weight_decay'] != 0:
                   d_p.add_(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['grad_prev'] = torch.zeros_like(p.data)
                else:
                    state['grad_prev'] = state['grad_prev'].type_as(p.data)
        
                grad_prev = state['grad_prev']

                grad_diff = torch.norm(d_p-grad_prev, p=2)
                x_diff = group['lr']*torch.norm(d_p, p=2)
                
                state['grad_prev'] = torch.clone(d_p).detach()
                
                eps = group['delta'] 
                
                momentum_val = max(0, min(1.-eps, ((1.-group['lr']*(grad_diff/x_diff))/(1.+group['lr']*(grad_diff/x_diff)))**2))

                if momentum_val != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum_val).add_(d_p, alpha=1.-momentum_val)
                    d_p = buf
                # Need to avoid version tracking for parameters
                p.data.add_(d_p, alpha=-group['lr'])

        return loss