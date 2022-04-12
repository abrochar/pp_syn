import numpy as np
import torch
import argparse
import scipy.io
import sys,os
sys.path.append(os.getcwd())
#sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from utils_gpu import pos_to_im3


# choose parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='voronoi')
parser.add_argument('--res', type=int, default=128)
parser.add_argument('--nGPU', type=int, default=1)
parser.add_argument('--dft', type=bool, default=False)
parser.add_argument('--sw', type=bool, default=False)
parser.add_argument('--aperiodic', default=False, action='store_true')
parser.add_argument('--normalize', default=False, action='store_true')
opts = parser.parse_args()


# to fix random seed
#torch.manual_seed(999)
#torch.cuda.manual_seed_all(999)
#torch.backends.cudnn.deterministic = True


# import the functions to compute the loss and the optimizer (L-BFGS)
if opts.nGPU == 1:
    from lbfgs2_routine import call_lbfgs2_routine, \
        obj_func_id, obj_func, dft
else :
    from lbfgs2_routine_par import call_lbfgs2_routine, \
        obj_func_id, obj_func, dft


# load the observation, in a square window of size 'res'
res = opts.res
if opts.data == 'voronoi':
    filename = './data/voronoi.txt'
    pos = res*np.loadtxt(filename, delimiter=',', skiprows=1, usecols=(1,2))
elif opts.data == 'turbulence':
    filename = './data/turbulence.txt'
    pos = res*np.loadtxt(fname=filename, delimiter=' ', skiprows=1, usecols=(1,2))/256
x_ = torch.from_numpy(pos).type(torch.float).cuda()
nb_points = pos.shape[0]


# define some constants necessary for discrete computation of the descriptor
res_ = torch.tensor(res).type(torch.float).cuda()
Mx_ =  torch.arange(0, res).type(torch.float).cuda()
My_ = torch.arange(0, res).type(torch.float).cuda()
pi_ = torch.from_numpy(np.array([np.pi])).float().cuda()


# Parameters for the transform
J = 4
L = 4
M, N = res, res
delta_j = 2
delta_l = L/2
delta_k = 0
delta_n = 1
nb_chunks = 3
nb_restarts = 0
nGPU = opts.nGPU
maxite = 100
factr = 1e3
# additional parameters for regularization
sw = opts.sw
nb_dir = 16
reg_par = 1e0

# wavelet phase harmonics moments w/o nomrlization, w/o periodicity

if opts.normalize and not opts.aperiodic:
    from kymatio.phase_harmonics_norm \
        import PhaseHarmonics2d
    print('periodic norm')
elif opts.normalize and opts.aperiodic:
    from kymatio.aperiodic_phase_harmonics_norm \
        import PhaseHarmonics2d
    print('aperiodic norm')
elif not opts.normalize and not opts.aperiodic:
    from kymatio.phase_harmonics \
        import PhaseHarmonics2d
    print('periodic')
else:
    from kymatio.aperiodic_phase_harmonics \
        import PhaseHarmonics2d
    print('aperiodic')

# streams are only needed for multi-gpu computations
wph_streams = []
for devid in range(nGPU):
    with torch.cuda.device(devid):
        s = torch.cuda.Stream()
        wph_streams.append(s)

# due to possible memory constraints, the set of descriptors
# is computed by parts (or 'chunks'), in order to free the memory
# after each part is computed


def build_obj(sigma):
    # build image from positions
    im = pos_to_im3(x_, res_, Mx_, My_, pi_, sigma)
    # list that will contain the values of the descriptors on the image
    Sims = []
    # dict that will contain the operator defining the descriptor
    wph_ops = dict()
    nCov = 0
    opid = 0
    for chunk_id in range(nb_chunks+1):
        devid = opid % nGPU
        wph_op = PhaseHarmonics2d(M, N, J, L, delta_j,
                                  delta_l, delta_k, delta_n,
                                  nb_chunks, chunk_id, devid)
        wph_op = wph_op.cuda()
        wph_ops[chunk_id] = wph_op
        im_ = im.to(devid)
        with torch.cuda.device(devid):
            torch.cuda.stream(wph_streams[devid])
            Sim_ = wph_op(im_)  # TO *factr it internally
            nCov += Sim_.shape[2]
            opid += 1
            Sims.append(Sim_)
    return wph_ops, Sims


# the following is needed only is the dft
# should be part of the descriptor

omegas_v = torch.arange(16).type(torch.double).cuda()/16*np.pi
thetas_v = torch.stack((torch.cos(omegas_v), torch.sin(omegas_v)), dim=0)


def shift(x):
    N = x.shape[0]
    x_ = torch.Tensor(x)
    x_ = torch.cat((x_[N//2:,:], x_[:N//2,:]), dim=0)
    xs = torch.cat((x_[:,N//2:], x_[:,:N//2]), dim=1)
    return xs.numpy()

if opts.dft:
    im_dft = dft(x_/res, res, 1)/nb_points

    range_ = np.arange(res)
    [y,x] = np.meshgrid(range_, range_)
    x=x-(res/2)
    y=y-(res/2)
    modx=torch.Tensor(shift(np.fft.fftshift(np.sqrt(x**2 + y**2)))).cuda()
    K = res//2
    masks = torch.zeros(K, res, res).cuda()
    for i in range(K):
        masks[i,...] = ((modx>=i) * (modx<(i+1)));

    Emask0 = im_dft.unsqueeze(0)*masks;
    ps_init = torch.log10(Emask0.sum(dim=-1).sum(dim=-1)/masks.sum(dim=-1).sum(dim=-1))
else:
    ps_init = 0
    masks = 0


# define the sequence of sigmas for multi-scale optim.

log_res = int(np.log2(res))
sigmas = [(log_res-3)/(2**s) for s in range(log_res-3)]


# compute norm of K for observation (at smallest sigma)
# and normalizing constant for relative energy

sigma = .5
wph_ops, Sims = build_obj(sigma)
K_obs = 0
for op_id in range(len(wph_ops)):
    ks = torch.mul(Sims[op_id], Sims[op_id]).sum()
    K_obs += ks


# perform syntheses ---

k = 0
while k<1:

    torch.cuda.synchronize()
    # sample initial configuration
    x0 = torch.torch.Tensor(nb_points, 2).uniform_(0,res).cuda()
    x0_ = x0.clone()
    # loop on the sequence of sigmas (for multi-scale)
    for i in range(len(sigmas)):
        if i==0:
            # compute K for the observation
            wph_ops, Sims = build_obj(sigmas[i])
            # perform synthesis
            x_fin = call_lbfgs2_routine(x0,sigmas[i],res,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,
                                        x0_, thetas_v, reg_par, sw, ps_init, masks, nGPU)
        else:
            # compute K for the observation
            wph_ops, Sims = build_obj(sigmas[i])
            # perform synthesis, starting from last optim results (with previous sigma)
            x_fin = call_lbfgs2_routine(x_fin,sigmas[i],res,wph_ops,wph_streams,Sims,nb_restarts,maxite,factr,
                                        x0_, thetas_v, reg_par, sw, ps_init, masks, nGPU)

    # uniform perturbation in a ball
#    perturb_ang = torch.Tensor(nb_points).uniform_(0,1).cuda()*2*np.pi
#    perturb_ang = torch.stack((torch.cos(perturb_ang), torch.sin(perturb_ang)), dim=1)
#    perturb_norm = torch.Tensor(nb_points).uniform_(0,1).cuda()
#    perturb = perturb_norm.unsqueeze(1)*perturb_ang

    # uniform perturbation in a square
#    perturb = torch.Tensor(nb_points, 2).uniform_(-.5,.5).cuda()

#    x_fin = x_fin + perturb

    if torch.isnan(x_fin.sum()).item():
        k = k-1
    else :
        ()  # save result if necessary
    k = k + 1


# compute relative energy of synthesis ---

def obj_func_id(x,wph_ops,Sims,sigma,res_,Mx,My,pi,op_id):
    wph_op = wph_ops[op_id]
    im = pos_to_im3(x, res, Mx, My, pi, sigma)
    p = wph_op(im)
    diff = p-Sims[op_id]
    loss = torch.mul(diff,diff).sum()
    return loss

def energy(x,wph_ops,Sims,sigma,res_,Mx,My,pi):
    loss = 0
    for op_id in range(len(wph_ops)):
        loss_t = obj_func_id(x,wph_ops,Sims,sigma,res_,Mx,My,pi,op_id)
        loss = loss + loss_t
    return loss

energy_syn = energy(x_fin,wph_ops,Sims,sigma,res_,Mx_,My_,pi_) / K_obs
print(energy_syn.item())



# Plot result ---

im = pos_to_im3(x_, res_, Mx_, My_, pi_, .5).squeeze().cpu()
min = np.percentile(im, 1)
max = np.percentile(im, 99)
plt.imshow(im, vmin=min, vmax=max);plt.figure()

im_fin = pos_to_im3(x_fin, res_, Mx_, My_, pi_, .5).detach().squeeze().cpu()
plt.imshow(im_fin, vmin=min, vmax=max); plt.show()


