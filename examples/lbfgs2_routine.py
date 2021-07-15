import os
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from utils_gpu import pos_to_im3, pos_to_im_chk
from tqdm import tqdm
gpu = True




def obj_func_id(x,wph_ops,Sims,factr2,sigma,res,Mx,My,pi,op_id):
    wph_op = wph_ops[op_id]
    im = pos_to_im3(x, res, Mx, My, pi, sigma)
    p = wph_op(im)
    diff = p-Sims[op_id]
    loss = torch.mul(diff,diff).sum()
    loss = loss*factr2
    return loss

def obj_func(x,wph_ops,Sims,factr2,sigma,res,Mx,My,pi):
    loss = 0
    if x.grad is not None:
        x.grad.data.zero_()
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id(x,wph_ops,Sims,factr2,sigma,res,Mx,My,pi,op_id)
        loss_t.backward() # accumulate grad into x.grad
        loss = loss + loss_t
    return loss


def sliced_wasserstein(x, x0_, thetas_v, reg_par=1):

    nb_points = x.shape[0]
    nb_dir = thetas_v.size(1)
    x_proj = torch.matmul(x, thetas_v)
    z = (torch.arange(nb_dir).repeat(nb_points)).cuda()
    x_proj_sorted_index = x_proj.argsort(dim=0).view(nb_points*nb_dir)*nb_dir+z
    x0_proj = torch.matmul(x0_, thetas_v)
    x0_proj_sorted_index = x0_proj.argsort(dim=0).view(nb_points*nb_dir)*nb_dir+z
    x_proj_sorted = torch.index_select(x_proj.view(nb_points*nb_dir), 0, x_proj_sorted_index)
    x0_proj_sorted = torch.index_select(x0_proj.view(nb_points*nb_dir), 0, x0_proj_sorted_index)
    diff = (x_proj_sorted - x0_proj_sorted).view(nb_points, nb_dir)
    loss = reg_par*torch.mul(diff, diff).mean()
    loss.backward(retain_graph=False)

    return loss


def pseudo_wasserstein(x, x0_, reg_par=1):

    diff = x - x0_
    loss = reg_par*torch.mul(diff, diff).sum(dim=-1).mean()
    loss.backward(retain_graph=False)

    return loss



def dft(pos, res, nb_chk):

    omegas0 = (torch.arange(res).type(torch.float).cuda()-res/2)*np.pi*2
    M0 = torch.cuda.FloatTensor(res, res, 2).fill_(0.)
    pts = pos.size(0)
    offset = 0
    M = M0.clone()

    for i in range(nb_chk):
        pos_ = pos[offset:offset+pts//nb_chk, :]
        nb_points = pos_.size(0)
        omegas = omegas0.clone().unsqueeze(0).repeat(nb_points, 1)
        pos_x = pos_[:, 0].unsqueeze(1).repeat(1, res)
        pos_y = pos_[:, 1].unsqueeze(1).repeat(1, res)
        prod_x = omegas*pos_x  # (nb_points, res)
        prod_y = omegas*pos_y  # (nb_points, res)
        prod = prod_x.unsqueeze(1).repeat(1, res, 1) + prod_y.unsqueeze(2).repeat(1, 1, res)  # (nb_points, res, res)
        exp = torch.stack((torch.cos(prod), -torch.sin(prod)), dim=-1).sum(0)  # (res, res, 2)
        M = M + exp
        offset += pts//nb_chk
    pos_ = pos[offset:, :]
    nb_points = pos_.size(0)
    omegas = omegas0.clone().unsqueeze(0).repeat(nb_points, 1)
    pos_x = pos_[:, 0].unsqueeze(1).repeat(1, res)
    pos_y = pos_[:, 1].unsqueeze(1).repeat(1, res)
    prod_x = omegas*pos_x  # (nb_points, res)
    prod_y = omegas*pos_y  # (nb_points, res)
    prod = prod_x.unsqueeze(1).repeat(1, res, 1) + prod_y.unsqueeze(2).repeat(1, 1, res)  # (nb_points, res, res)
    exp = torch.stack((torch.cos(prod), -torch.sin(prod)), dim=-1).sum(0)  # (res, res, 2)
    M = M + exp
    M = M.norm(dim=-1)**2

    return M

def dft_loss(x, res, nb_chk, ps0, masks):
    dft_current = dft(x/res, res, nb_chk)/x.shape[0]
    Emask = dft_current.unsqueeze(0)*masks;
    ps = torch.log10(Emask.sum(dim=-1).sum(dim=-1)/masks.sum(dim=-1).sum(dim=-1))
    diff = ps - ps0
#    diff = torch.log10(dft_current) - torch.log10(dft0)
    loss = torch.mul(diff, diff).mean()
    loss.backward(retain_graph=False)

    return loss


def call_lbfgs2_routine(x0,sigma,res,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,\
                        x0_,thetas_v, reg_par, sw, ps_init, masks,\
                        nGPU=2, maxcor=100,gtol=1e-10,ftol=1e-10):

    res_ = torch.tensor(res).type(torch.float).cuda()
    Mx_ =  torch.arange(0, res).type(torch.float).cuda()
    My_ = torch.arange(0, res).type(torch.float).cuda()
    pi_ = torch.from_numpy(np.array([np.pi])).float().cuda()


    # x0 init points (no need to be on GPU)
    # sigma: gaussian width
    # return x: optimal points
    for start in range(nb_restarts+1):
        if start==0:
            x = x0.cuda()
            x.requires_grad_(True)
        time0 = time()
        optimizer = optim.LBFGS({x}, max_iter=maxite, line_search_fn='strong_wolfe',\
                                tolerance_grad = gtol, tolerance_change = ftol,\
                                history_size = maxcor)
            
        def closure():
            optimizer.zero_grad()
            loss = obj_func(x,wph_ops,Sims,factr**2,sigma,res_,Mx_,My_,pi_)
            pbar.update(1)
            return loss

        pbar = tqdm(total = maxite)
        optimizer.step(closure)
        pbar.close()

        opt_state = optimizer.state[optimizer._params[0]]
        niter = opt_state['n_iter']
        final_loss = opt_state['prev_loss']

        print('At restart',start,'OPT fini avec:', final_loss,niter,'in',time()-time0,'sec')
                
    return x
