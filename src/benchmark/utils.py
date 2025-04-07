import math
import torch

import crypten
import crypten.communicator as comm
import crypten.nn as cnn

from crypten.mpc.primitives.converters import convert
from crypten.mpc.mpc import MPCTensor
from crypten.mpc.ptype import ptype as Ptype

def encrypt_tensor(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    assert world_size  == 2
    
    # assumes party 1 is the actual data provider
    src_id = 1

    if rank == src_id:
        input_upd = input#.cuda()
    else:
        input_upd = torch.empty(input.size())#.cuda()
    private_input = crypten.cryptensor(input_upd, src=src_id)
#    print(private_input)
    return private_input

def encrypt_model(model, modelFunc, config, dummy_input):
    rank = comm.get().get_rank()
    
    # assumes party 0 is the actual model provider
    if rank == 0:
        model_upd = model #.cuda()
    else:
        if isinstance(config, tuple):
            model_upd = modelFunc(config[0], config[1]) #.cuda()
        else:
            model_upd = modelFunc(config) #.cuda()

    private_model = model_upd.encrypt(src=0)
    return private_model


class softmax_2RELU(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = cnn.ReLU()
        self.dim = dim

    def forward(self, x):
        func_x = self.func(x)
        return func_x / func_x.sum(keepdim=True, dim=self.dim)

class softmax_2QUAD(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def forward(self, x):
        a, b, c, d = x.size()
        #quad = x#self.norm(x)
        quad = (x+5) * (x+5)
        return quad / quad.sum(dim=self.dim, keepdims=True)

class activation_newGeLU(cnn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.three = torch.tensor([3.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.pi_const = torch.tensor([math.sqrt(2/math.pi)]).item()
        self.pow = cnn.Pow()
        self.tanh = cnn.Hardtanh()

    def forward(self, x):
        return self.half * x * (self.one + self.tanh(self.pi_const * (x + self.constant * self.pow((x, self.three)))))


class activation_quad(cnn.Module):
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef
        #return x*x

""" PUMA SPECIFIC ACTIVATION FUNCTIONS """


class softmax_PUMA(cnn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.lim = -14

    def forward(self, x):
        # Ensure maximum element is slightly less than 0.
        x_max = x.max(dim=-1, keepdim=True, one_hot=False)[0]
        x = x - x_max - self.eps

        # Large enough negative clipped to zero.
        b = x > self.lim

        # Compute nonzero exponential contributions. Normalize them.
        numer = b * x.exp()
        denom = numer.sum(dim=-1, keepdim=True)
        return numer / denom


class activation_newGeLU_PUMA(cnn.Module):
    def __init__(self):
        super().__init__()

        # Piecewise polynomial limits.
        self.lim = [-4.0, -1.95, 3.0]

        # Polynomial weights for two of the three limits.
        self.aw = [-0.5054031199708174, -0.42226581151983866, -0.11807612951181953, -0.011034134030615728]
        self.bw = [0.008526321541038084, 0.5, 0.3603292692789629, 0.0, -0.037688200365904236, 0.0, 0.0018067462606141187]

    def forward(self, x):
        # Indicators for three (not disjoint) intervals forming the piecewise approximation.
        b0 = x < self.lim[0]
        b1 = x < self.lim[1]
        b2 = x > self.lim[2]

        # Must modify to binary primitives to allow for XOR, to get indicators for disjoint intervals.
        b0t = convert(b0._tensor, Ptype.binary)
        b1t = convert(b1._tensor, Ptype.binary)
        b2t = convert(b2._tensor, Ptype.binary)
        b3 = b1t.__xor__(b2t).__xor__(True)         # is x in [-1.95, 3.0]
        b4 = b0t.__xor__(b1t)                       # is x in [-4, -1.95]

        # Back to arithmetic secret sharing for the needed ones.
        b3 = convert(b3, Ptype.arithmetic)
        b4 = convert(b4, Ptype.arithmetic)

        # Powers of x, with square for efficiency.
        x2 = x.square()
        x3 = x * x2
        x4 = x2.square()
        x6 = x3.square()

        # Compute values of polynomial segments.
        seg1 = self.aw[3] * x3 + self.aw[2] * x2 + self.aw[1] * x + self.aw[0]
        seg2 = self.bw[6] * x6 + self.bw[4] * x4 + self.bw[2] * x2 + self.bw[1] * x + self.bw[0]

        # Must make interval indicators back to MPCTensor with arithmetic sharing before executing final weighted sum.
        b3 = MPCTensor.from_shares(b3._tensor)
        b4 = MPCTensor.from_shares(b4._tensor)
        ret = b2 * x +  b4 * seg1 + b3 * seg2
        return ret
