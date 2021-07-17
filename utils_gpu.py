import numpy as np
import torch
#import matplotlib.pyplot as plt

def weight(index, pos, res, pi, sigma):
    # index: (1,res)
    # pos: (nbp,1)
    # res, pi, sigma: number
    v = torch.min(torch.min(torch.abs(pos-index), torch.abs(pos+res-index)),
                  torch.abs(pos-res-index))
    w = torch.exp(-torch.mul(v,v)/(2*(sigma**2)))/torch.sqrt(2*pi*sigma**2)
    return w

def pos_to_im3(x, res, Mx, My, pi, sigma):
    # x: (nbp, 2)
    # Mx: (res), My: (res)
    im_x = weight(Mx.unsqueeze(0), x[:, 0].unsqueeze(1), res, pi, sigma).unsqueeze(2)
    im_y = weight(My.unsqueeze(0), x[:, 1].unsqueeze(1), res, pi, sigma).unsqueeze(1)
    M = torch.matmul(im_x, im_y).sum(0)
    return M.unsqueeze(0).unsqueeze(0)


def pos_to_im_chk(xp, res_, Mx_, My_, pi_, sigma, k, op_id, nGPU):
    offset = 0
    devid = op_id % nGPU
    nb_points = xp.size(0)
    im = torch.zeros(1,1,size,size).type(torch.float).cuda()
    im = im.to(devid)
    for i in range(k):
        x_chk = xp[offset:offset+nb_points//k, :]
        im = im + pos_to_im3(x_chk, res_, Mx_, My_, pi_, sigma)
        offset += nb_points//k
    x_chk = xp[offset:, :]
    im = im + pos_to_im3(x_chk, res_, Mx_, My_, pi_, sigma)
    return im
