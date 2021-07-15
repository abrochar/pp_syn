import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReflectionPad2d
from torch.nn import ZeroPad2d
from torch.autograd import Function
import numpy as np


def shift(x):
    N = x.shape[0]
    x_ = torch.Tensor(x)
    x_ = torch.cat((x_[N//2:,:,:], x_[:N//2,:,:]), dim=0)
    xs = torch.cat((x_[:,N//2:,:], x_[:,:N//2,:]), dim=1)
    return xs

def dft(pos, res, nb_chunks, om):
    pts = pos.size(0)
    offset = 0

    for i in range(nb_chunks):
        pos_ = pos[offset:offset+pts//nb_chunks, :]
        nb_points = pos_.size(0)
        omegas = om.unsqueeze(0).repeat(nb_points, 1)
        pos_x = pos_[:, 0].unsqueeze(1).repeat(1, res)
        pos_y = pos_[:, 1].unsqueeze(1).repeat(1, res)

        prod_x = omegas*pos_x  # (nb_points, res)
        prod_y = omegas*pos_y  # (nb_points, res)

        prod = prod_x.unsqueeze(1).repeat(1, res, 1) + prod_y.unsqueeze(2).repeat(1, 1, res)  # (nb_points, res, res)

        exp = torch.stack((torch.cos(prod), -torch.sin(prod)), dim=-1).sum(0)  # (res, res, 2)

        if i==0:
            M = exp
        else:
            M = M + exp

        offset += pts//nb_chunks

    pos_ = pos[offset:, :]
    nb_points = pos_.size(0)
    omegas = om.unsqueeze(0).repeat(nb_points, 1)
    pos_x = pos_[:, 0].unsqueeze(1).repeat(1, res)
    pos_y = pos_[:, 1].unsqueeze(1).repeat(1, res)

    prod_x = omegas*pos_x  # (nb_points, res)
    prod_y = omegas*pos_y  # (nb_points, res)

    prod = prod_x.unsqueeze(1).repeat(1, res, 1) + prod_y.unsqueeze(2).repeat(1, 1, res)  # (nb_points, res, res)

    exp = torch.stack((torch.cos(prod), -torch.sin(prod)), dim=-1).sum(0)  # (res, res, 2)

    M = M + exp
    M = M.norm(dim=-1)**2

    return M




def maskns(J, M, N):
    m = torch.ones(J, M, N)
    for j in range(J):
        for x in range(M):
            for y in range(N):
                if (x<(2**j)//2 or y<(2**j)//2 \
                or x+1>M-(2**j)//2 or y+1>N-(2**j)//2):
                    m[j, x, y] = 0
    m = m.type(torch.float)
    m = m / m.sum(dim=(-1,-2), keepdim=True)
    m = m*M*N
    return m


def maskns_o2(J, M, N):
    m = torch.ones(2**J, M, N)
    for s in range(2**J):
        for x in range(M):
            for y in range(N):
                if (x<s-1 or y<s-1 \
                or x>M-s or y>N-s):
                    m[s, x, y] = 0
    m = m.type(torch.float)
    m = m / m.sum(dim=(-1,-2), keepdim=True)
    m = m*M*N
    return m

def masks_subsample_shift(J,M,N):
    m = torch.zeros(M,N).type(torch.float)
    m[0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            m[x,y] = 1.
    return m

def masks_subsample_shift2(J,M,N):
    m = torch.zeros(J,M,N).type(torch.float)
    m[:,0,0] = 1.
    angles = torch.arange(8).type(torch.float)
    angles = angles/8*2*np.pi
    for j in range(J):
        for theta in range(8):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            m[j,x,y] = 1.
    return m

def iscomplex(input):
    return input.size(-1) == 2


def ones_like(z):
    re = torch.ones_like(z[..., 0])
    im = torch.zeros_like(z[..., 1])
    return torch.stack((re, im), dim=-1)

def real(z):
    z_ = z.clone()
    return z_[..., 0]


def imag(z):
    z_ = z.clone()
    return z_[..., 1]


def conjugate(z):
    z_copy = z.clone()
    z_copy[..., 1] = -z_copy[..., 1]
    return z_copy


def pows(z, max_k, dim=0):
    z_pows = [ones_like(z)]
    if max_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_k + 1):
            z_acc = mul(z_acc, z)
            z_pows.append(z_acc)
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows


def log2_pows(z, max_pow_k, dim=0):
    z_pows = [ones_like(z)]
    if max_pow_k > 0:
        z_pows.append(z)
        z_acc = z
        for k in range(2, max_pow_k + 1):
            z_acc = mul(z_acc, z_acc)
            z_pows.append(z_acc)
    assert len(z_pows) == max_pow_k + 1
    z_pows = torch.stack(z_pows, dim=dim)
    return z_pows

def mul(z1_, z2_):
    z1 = z1_.clone()
    z2 = z2_.clone()
    zr = real(z1) * real(z2) - imag(z1) * imag(z2)
    zi = real(z1) * imag(z2) + imag(z1) * real(z2)
    z = z1.new(z1.size())
    z[...,0] = zr
    z[...,1] = zi
    return z

# substract spatial mean (complex valued input)
class SubInitSpatialMeanC(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output

class SubInitSpatialMeanCrl(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach() 
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output


# substract spatial mean (real valued input)
class SubInitSpatialMeanR(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, -1, True)
            minput = torch.mean(minput, -2, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output

class SubInitMeanIso(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            minput = torch.mean(minput, -2, True)
            minput = torch.mean(minput, -3, True)
            minput[:, 1:, ...] = 0
            self.minput = minput
#            print('sum of minput', self.minput.sum())
            # print('minput shape is', self.minput.shape)
        output = input - self.minput
        return output

class DivInitStd(object):
    def __init__(self,stdcut=1e-6):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-2]*input.shape[-3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=-2, keepdim=True).norm(dim=-3, keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
#        print(self.stdinput)
        return output

class DivInitStdrl(object):
    def __init__(self,stdcut=1e-6):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-2]*input.shape[-3]
            #stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=-2, keepdim=True).norm(dim=-3, keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
#        print(self.stdinput)
        return output

class DivInitStdR(object):
    def __init__(self,stdcut=0):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(...,M,N)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-1]*input.shape[-2]
            stdinput = torch.norm(stdinput, dim=(-2,-1), keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
        return output

class DivInitStdRot(object):
    def __init__(self,stdcut=0):
        self.stdinput = None
        self.eps = stdcut
#        print('DivInitStd:stdcut',stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(1,P_c,M,N,2)
            m = torch.mean(torch.mean(stdinput, -2, True), -3, True)
            stdinput = stdinput - m
            d = input.shape[-2]*input.shape[-3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=(-2, -3), keepdim=True)
            self.stdinput = stdinput  / np.sqrt(d)
            self.stdinput = self.stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
        return output



class DivInitMax(object):
    def __init__(self):
        self.max = None

    def __call__(self, input):
        if self.max is None:
            maxinput = input.clone().detach()  # input size:(1,P_c,M,N)
            maxinput = torch.max(maxinput, dim=-1, keepdim=True)[0]
            maxinput = torch.max(maxinput, dim=-2, keepdim=True)[0]
            self.max = maxinput
        output = input/self.max

        return output


class DivInitMean(object):
    def __init__(self):
        self.mean = None

    def __call__(self, input):
        if self.mean is None:
            if input.size(-1) > 2:
                meaninput = input.clone().detach()  # input size:(1,P_c,M,N)
                meaninput = torch.mean(meaninput, dim=-1, keepdim=True)[0]
                meaninput = torch.mean(meaninput, dim=-2, keepdim=True)[0]
                self.mean = meaninput+1e-6
            else:
                meaninput = input.clone().detach()  # input size:(1,P_c,M,N,2)
                meaninput = torch.mean(meaninput, dim=-2, keepdim=True)
                meaninput = torch.mean(meaninput, dim=-3, keepdim=True)
                self.mean = meaninput

        output = input/self.mean

        return output


class DivInitStdQ0(object):
    def __init__(self):
        self.stdinput = None
        self.eps = 1e-16

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(J,Q,K,M,N,2)
            stdinput = stdinput[:, :1, ...]  # size:(J,1,K,M,N,2)
            d = input.shape[-2]*input.shape[-3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True)
            stdinput = torch.norm(stdinput, dim=(-2, -3), keepdim=True) / np.sqrt(d)
            self.stdinput = stdinput + self.eps
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())
        output = input/self.stdinput
        return output

# substract spatial mean (complex valued input), average over ell
class SubInitSpatialMeanCL(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input): # input: (J,L2,M,N,K,2)
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, dim=1, keepdim=True)
            minput = torch.mean(minput, dim=2, keepdim=True)
            minput = torch.mean(minput, dim=3, keepdim=True)
            self.minput = minput # .expand_as(input)
#            print('minput size',self.minput.shape)
#            print('sum of minput',self.minput.sum())

        output = input - self.minput
        return output

# divide by std, average over ell
class DivInitStdL(object):
    def __init__(self):
        self.stdinput = None

    def __call__(self, input): # input: (J,L2,M,N,K,2)
        if self.stdinput is None:
            stdinput = input.clone().detach()
            #dl = input.shape[1]*input.shape[2]*input.shape[3]
            stdinput = torch.norm(stdinput, dim=-1, keepdim=True) # (J,L2,M,N,K,1)
            stdinput = torch.mul(stdinput,stdinput)
            stdinput = torch.mean(stdinput, dim=1, keepdim=True)
            stdinput = torch.mean(stdinput, dim=2, keepdim=True)
            stdinput = torch.mean(stdinput, dim=3, keepdim=True)
            self.stdinput = torch.sqrt(stdinput) # .expand_as(input) #  / dl)
#            print('stdinput size',self.stdinput.shape)
#            print('stdinput max,min:',self.stdinput.max(),self.stdinput.min())

        output = input/self.stdinput
        return output

class SubInitSpatialMeanCinFFT(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = input[...,0,0,:] # zero-freq. value torch.mean(minput, -2, True)
            #minput = torch.mean(minput, -3, True)
            self.minput = minput
#            print('sum of minput',self.minput.sum())

        output = input
        output[...,0,0,:] = input[...,0,0,:] - self.minput
        return output

class SubsampleFourier(object):
    """
        Subsampling of a 2D image performed in the Fourier domain
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.
        Parameters
        ----------
        x : tensor_like
            input tensor with at least 5 dimensions, the last being the real
             and imaginary parts.
            Ideally, the last dimension should be a power of 2 to avoid errors.
        k : int
            integer such that x is subsampled by 2**k along the spatial variables.
        Returns
        -------
        res : tensor_like
            tensor such that its fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            FFT^{-1}(res)[u1, u2] = FFT^{-1}(x)[u1 * (2**k), u2 * (2**k)]
    """
    def __call__(self, input, k):
        if input.ndim == 5:
            out = input.new(input.size(0), input.size(1), input.size(2) // k, input.size(3) // k, 2)


            y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)

            out = y.mean(4, keepdim=False).mean(2, keepdim=False)
        elif input.ndim==4:
            out = input.new(input.size(0), input.size(1) // k, input.size(2) // k, 2)


            y = input.view(input.size(0),
                           input.size(1)//out.size(1), out.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           2)

            out = y.mean(3, keepdim=False).mean(1, keepdim=False)
        else:  #ndim=3
            out = input.new(input.size(0) // k, input.size(1) // k, 2)


            y = input.view(
                           input.size(0)//out.size(0), out.size(0),
                           input.size(1)//out.size(1), out.size(1),
                           2)

            out = y.mean(2, keepdim=False).mean(0, keepdim=False)

        return out

class SubsampleFourier2(object):

    def __call__(self, input, k):

        k_ = 2*k
        # ndim = 5
        out = input.new(input.size(0), input.size(1), input.size(2) // k_, input.size(3) // k_, 2)

        y = input.view(input.size(0), input.size(1),
                           input.size(2)//out.size(2), out.size(2),
                           input.size(3)//out.size(3), out.size(3),
                           2)
        s = [0,-1]
        out = y[:,:,s,:,s,:,:]

        return out

class SubInitMean(object):
    def __init__(self, dim):
        self.dim = dim # use the last "dim" dimensions to compute the mean
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            #print('subinitmean:input',input.shape)
            for d in range(self.dim):
                minput = torch.mean(minput, -1)
            for d in range(self.dim):
                minput = minput.unsqueeze(-1)
            #print('subinitmean:minput',minput.shape)
            minput.expand_as(input)
            self.minput = minput

        #print('subinitmean:minput sum',self.minput.sum())
        output = input - self.minput
        return output

class Pad(object):
    def __init__(self, pad_size, pre_pad=False, pad_mode='zero'):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : int
                size of padding to apply.
            pre_pad : boolean
                if set to true, then there is no padding, one simply adds the imaginarty part.
        """
        self.pre_pad = pre_pad
        if pad_mode == 'Reflect':
#            print('use reflect pad')
            self.padding_module = ReflectionPad2d(pad_size)
        else:
#            print('use zero pad')
            self.padding_module = ZeroPad2d(pad_size)

    def __call__(self, input):
        if(self.pre_pad):
            output = input.new_zeros(input.size(0), input.size(1), input.size(2), input.size(3), 2)
            output.narrow(output.ndimension()-1, 0, 1)[:] = input
        else:
            out_ = self.padding_module(input)
            output = input.new_zeros(*(out_.size() + (2,)))
            output.select(4, 0)[:] = out_

        return output



def padc(x):
    x_ = x.clone()
    return torch.stack((x_, torch.zeros_like(x_)), dim=-1)




def unpad(in_):
    """
        Slices the input tensor at indices between 1::-1
        Parameters
        ----------
        in_ : tensor_like
            input tensor
        Returns
        -------
        in_[..., 1:-1, 1:-1]
    """
    return in_[..., 1:-1, 1:-1]

class Modulus(object):
    """
        This class implements a modulus transform for complex numbers.
        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)
        Parameters
        ---------
        x: input tensor, with last dimension = 2 for complex numbers
        Returns
        -------
        output: a tensor with imaginary part set to 0, real part set equal to
        the modulus of x.
    """
    def __call__(self, input):

        norm = input.norm(p=2, dim=-1, keepdim=True)
        return torch.cat([norm, torch.zeros_like(norm)], -1)


def modulus(z):
    z_mod = z.norm(p=2, dim=-1)

    # if z.requires_grad:
    #     # z_mod_mask.register_hook(HookDetectNan("z_mod_mask in modulus"))
    #     z_mod.register_hook(HookDetectNan("z_mod in modulus"))
    #     z.register_hook(HookDetectNan("z in modulus"))

    return z_mod


def fft(input, direction='C2C', inverse=False):
    """
        Interface with torch FFT routines for 2D signals.
        Example
        -------
        x = torch.randn(128, 32, 32, 2)
        x_fft = fft(x, inverse=True)
        Parameters
        ----------
        input : tensor
            complex input for the FFT
        direction : string
            'C2R' for complex to real, 'C2C' for complex to complex
        inverse : bool
            True for computing the inverse FFT.
            NB : if direction is equal to 'C2R', then the transform
            is automatically inverse.
    """
    if direction == 'C2R':
        inverse = True

    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

    if (not input.is_contiguous()):
        raise (RuntimeError('Tensors must be contiguous!'))

    if direction == 'C2R':
        output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)
    elif direction == 'C2C':
        if inverse:
            output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
        else:
            output = torch.fft(input, 2, normalized=False)

    return output



class PhaseHarmonics2(Function):
    @staticmethod
    def forward(ctx, z, k):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)
        theta = torch.atan2(y, x)
        ktheta = k * theta
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)
        ctx.save_for_backward(x, y, r, k)
        return r.unsqueeze(-1)*eiktheta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, k = ctx.saved_tensors
        theta = torch.atan2(y, x)
        ktheta = k * theta
        cosktheta = torch.cos(ktheta)
        sinktheta = torch.sin(ktheta)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        df1dx = costheta*cosktheta + k*sintheta*sinktheta
        df2dx = costheta*sinktheta - k*sintheta*cosktheta
        df1dy = sintheta*cosktheta - k*costheta*sinktheta
        df2dy = sintheta*sinktheta + k*costheta*cosktheta

        dx1 = df1dx*grad_output[...,0] + df2dx*grad_output[...,1]
        dx2 = df1dy*grad_output[...,0] + df2dy*grad_output[...,1]

        return torch.stack((dx1, dx2), -1), k # dummy gradient torch.zeros_like(k)

class PhaseHarmonicsIso(Function):
    # z.size(): (J,L2,M,N,1,2)
    # k.size(): (K)
    @staticmethod
    def forward(ctx, z, k):
        z = z.detach()
        x, y = real(z), imag(z)
        r = z.norm(p=2, dim=-1)  # (J, L2, M, N, 1)
        theta = torch.atan2(y, x)  # (J, L2, M, N, 1)
#        print(theta.size(), k.size())
        ktheta = k * theta  # (J, L2, M, N, K)
        eiktheta = torch.stack((torch.cos(ktheta), torch.sin(ktheta)), dim=-1)
        # eiktheta.size(): (J, L2, M, N, K, 2)
        ctx.save_for_backward(x, y, r, k)
        return r.unsqueeze(-1)*eiktheta

    @staticmethod
    def backward(ctx, grad_output):
        x, y, r, k = ctx.saved_tensors
        theta = torch.atan2(y, x)
        ktheta = k * theta
        cosktheta = torch.cos(ktheta)
        sinktheta = torch.sin(ktheta)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        df1dx = costheta*cosktheta + k*sintheta*sinktheta
        df2dx = costheta*sinktheta - k*sintheta*cosktheta
        df1dy = sintheta*cosktheta - k*costheta*sinktheta
        df2dy = sintheta*sinktheta + k*costheta*cosktheta

        dx1 = df1dx*grad_output[..., 0] + df2dx*grad_output[..., 1]
        dx2 = df1dy*grad_output[..., 0] + df2dy*grad_output[..., 1]

        return torch.stack((dx1, dx2), -1), torch.zeros_like(k)


# periodic shift in 2d

class PeriodicShift2D(nn.Module):
    def __init__(self, M,N,shift1,shift2):
        super(PeriodicShift2D, self).__init__()
        self.M = M
        self.N = N
        self.shift1 = shift1 % M # [0,M-1]
        self.shift2 = shift2 % N # [0,N-1]

    def forward(self, input):
        # input dim is (1,P_c,M,N,2)
        # per. shift along M and N dim by shift1 and shift2
        #M = input.shape[2]
        #N = input.shape[3]
        M = self.M
        N = self.N
        shift1 = self.shift1
        shift2 = self.shift2

        #blk11 = [[0,0],[shift1-1,shift2-1]]
        #blk22 = [[shift1,shift2],[M-1,N-1]]
        #blk12 = [[shift1,0],[M-1,shift2-1]]
        #blk21 = [[0,shift2],[shift1-1,N-1]]
        output = input.clone()
        output[:,:,0:M-shift1,0:N-shift2,:] = input[:,:,shift1:M,shift2:N,:]
        output[:,:,0:M-shift1,N-shift2:N,:] = input[:,:,shift1:M,0:shift2,:]
        output[:,:,M-shift1:M,0:N-shift2,:] = input[:,:,0:shift1,shift2:N,:]
        output[:,:,M-shift1:M,N-shift2:N,:] = input[:,:,0:shift1,0:shift2,:]

        return output


def complex_mul(a, b):
    ar = a[..., 0]
    br = b[..., 0]
    ai = a[..., 1]
    bi = b[..., 1]
    real = ar*br - ai*bi
    imag = ar*bi + ai*br

    return torch.stack((real, imag), dim=-1)



def shift_filt(im_, u):
    # u: (J, L2, 2)
    # im_: (J, l2, M, N, 2)
    size = im_.size(-2)
    u = u.type(torch.float)
    map = torch.arange(size, dtype = torch.float).repeat(tuple(u.size()[:2])+(1,))  # (J, L2, N)
    z = torch.matmul(map.unsqueeze(-1), u.unsqueeze(-2))  # (J, L2, N, 1), (J, L2, 1, 2)->(J, L2, N, 2)
    sp = z[..., 0].unsqueeze(-1).repeat(1,1,1,size) + z[..., 1].unsqueeze(-2).repeat(1,1,size,1)  # (J, L2, N, N)
    del(z)
    # compute e^(-iw.u0)
    fft_shift = torch.stack((torch.cos(2*np.pi*sp/size), torch.sin(2*np.pi*sp/size)), dim=-1)  # (J, L2, N, N, 2)
    im_shift_fft = mul(fft_shift, im_)  # (J, L2, N, N, 2)
    del(fft_shift); del(sp)
    return im_shift_fft



