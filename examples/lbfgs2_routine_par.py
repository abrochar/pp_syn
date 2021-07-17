import os
import matplotlib.pyplot as plt
from time import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from utils_gpu import pos_to_im3, pos_to_im_chk
from tqdm import tqdm

torch.backends.cudnn.deterministic = True


def obj_func_id(x_t,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,op_id,nGPU):
    # convert points to im on devid, using a loop
    devid = op_id % nGPU
    res_t = ress[devid]
    Mx_t = Mxs[devid]
    My_t = Mys[devid]
    pi_t = pis[devid]
    with torch.cuda.device(devid):
        torch.cuda.stream(wph_streams[devid])
#        im_t = pos_to_im_chk(x_t, res_t, Mx_t, My_t, pi_t, sigma, 4, op_id, nGPU)
        im_t = pos_to_im3(x_t, res_t, Mx_t, My_t, pi_t, sigma)
        # compute wph grad on devid
        wph_op = wph_ops[op_id]
        p = wph_op(im_t)
#        diff = p/(Sims[op_id]+1e-9)-1
        diff = p - Sims[op_id]
        loss = torch.mul(diff,diff).sum()
        loss = loss*factr2

    return loss


def obj_func(x,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,nGPU):
    loss = 0
    loss_a = []

    # copy x to multiple gpus
    x_a = []
    for devid in range(nGPU):
        x_t = x.to(devid)
        x_a.append(x_t)

    # compute gradients with respect to x_a
    for op_id in range(len(wph_ops)):
        devid = op_id % nGPU
        x_t = x_a[devid]
        #print(x_t)
        loss_t = obj_func_id(x_t,wph_ops,wph_streams,Sims,factr2,sigma,ress,Mxs,Mys,pis,op_id,nGPU)
        loss_t.backward(retain_graph=False) # accumulate grad into x.grad
        loss_a.append(loss_t)

    torch.cuda.synchronize()

    # sum the loss
    for op_id in range(len(wph_ops)):
        loss = loss + loss_a[op_id].item()

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
    # x0 init points (no need to be on GPU)
    # sigma: gaussian width
    # return x: optimal points
    assert(nGPU >= 2)
    ress = []
    Mxs = []
    Mys = []
    pis = []
    x_a = []
    for devid in range(nGPU):
        res_ = torch.tensor(res).type(torch.float).cuda().to(devid)
        Mx_ =  torch.arange(0, res).type(torch.float).cuda().to(devid)
        My_ = torch.arange(0, res).type(torch.float).cuda().to(devid)
        pi_ = torch.from_numpy(np.array([np.pi])).float().cuda().to(devid)
        ress.append(res_)
        Mxs.append(Mx_)
        Mys.append(My_)
        pis.append(pi_)
    
    for start in range(nb_restarts+1):
        if start==0:
            x = x0.cuda()
            x.requires_grad_(True)
        time0 = time()
        optimizer = optim.LBFGS({x}, max_iter=maxite, line_search_fn='strong_wolfe',\
                                tolerance_grad = gtol, tolerance_change = ftol,\
                                history_size = maxcor)

#        optimizer = optim.SGD({x}, lr=1e0)

        
        def closure():
            optimizer.zero_grad()
            if sigma > 0:

                loss = obj_func(x,wph_ops,wph_streams,Sims,factr**2,sigma,ress,Mxs,Mys,pis,nGPU)
#                loss = loss + dft_loss(x, res, 1, ps_init, masks)  # for dft+wph
#                print(loss)
                if sw:
#                    loss = loss + sliced_wasserstein(x, x0_, thetas_v, reg_par)
                    loss = loss + pseudo_wasserstein(x, x0_, reg_par)
            else:
         #       torch.cuda.empty_cache()
                loss = dft_loss(x, x.size(-1), 1, ps_init, masks)
#                print(loss)
            pbar.update(1)
            return loss

        pbar = tqdm(total = maxite)
        optimizer.step(closure)
        pbar.close()

#        for it in range(maxite):
 #           optimizer.step(closure)

        opt_state = optimizer.state[optimizer._params[0]]
        niter = opt_state['n_iter']
        final_loss = opt_state['prev_loss']
        print('At restart',start,'OPT fini avec:', final_loss,niter,'in',time()-time0,'sec')
        
    return x
