import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    def __init__(self, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, ))
        self.scale_inv = nn.Parameter(torch.ones(1, ))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input, inv_init=False):
        with torch.no_grad():
            mean = input.mean().reshape((1, ))
            std = input.std().reshape((1, ))

            if inv_init:
                self.loc.data.copy_(torch.zeros_like(mean))
                self.scale_inv.data.copy_(torch.ones_like(std))
            else:
                self.loc.data.copy_(-mean)
                self.scale_inv.data.copy_((std + 1e-6))

    def forward(self, input):
        _, in_dim = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = -logabs(self.scale_inv)

        logdet = in_dim * torch.sum(log_abs)

        if self.logdet:
            return (1.0 / self.scale_inv) * (input + self.loc), logdet

        else:
            return (1.0 / self.scale_inv) * (input + self.loc)

    def reverse(self, output):
        _, in_dim = output.shape

        if self.initialized.item() == 0:
            self.initialize(output, inv_init=True)
            self.initialized.fill_(1)

        log_abs = -logabs(self.scale_inv)

        logdet = -in_dim * torch.sum(log_abs)

        if self.logdet:
            return output * self.scale_inv - self.loc, logdet

        else:
            return output * self.scale_inv - self.loc



class ZeroFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(out_dim, ))

    def forward(self, input):
        out = self.fc(input)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, ndim, seqfrac=4, affine=True):
        super().__init__()

        self.affine = affine

        # self.net = nn.Sequential(
        # 	nn.Linear(ndim-ndim//2, ndim // (2*seqfrac)),
        # 	nn.LeakyReLU(negative_slope=0.01, inplace=True),
        # 	nn.BatchNorm1d(ndim // (2*seqfrac)),
        # 	nn.Linear(ndim // (2*seqfrac), ndim // (2*seqfrac)),
        # 	nn.LeakyReLU(negative_slope=0.01, inplace=True),
        # 	nn.BatchNorm1d(ndim // (2*seqfrac)),
        # 	ZeroFC(ndim // (2*seqfrac), 2*(ndim // 2) if self.affine else ndim // 2),
        # )

        # self.net = nn.Sequential(
        #     nn.Linear(ndim-ndim//2, int(ndim / (2*seqfrac))),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.LayerNorm(int(ndim / (2*seqfrac))),
        #     nn.Linear(int(ndim / (2*seqfrac)), int(ndim / (2*seqfrac))),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.LayerNorm(int(ndim / (2*seqfrac))),
        #     ZeroFC(int(ndim / (2*seqfrac)), 2*(ndim // 2) if self.affine else ndim // 2),
        # )

        # older version has skip connection, but we find that not necessary
        self.net = nn.Sequential(
            nn.Linear(ndim-ndim//2, int(ndim / (2*seqfrac))),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Softplus(beta=1, threshold=20),
            # nn.Tanh(),
            nn.BatchNorm1d(int(ndim / (2*seqfrac)), eps=1e-2, affine=True),
            nn.Linear(int(ndim / (2*seqfrac)), int(ndim / (2*seqfrac))),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.Softplus(beta=1, threshold=20),
            # nn.Tanh(),
            nn.BatchNorm1d(int(ndim / (2*seqfrac)), eps=1e-2, affine=True),
            ZeroFC(int(ndim / (2*seqfrac)), 2*(ndim // 2) if self.affine else ndim // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[3].weight.data.normal_(0, 0.05)
        self.net[3].bias.data.zero_()
        # self.net[2].weight.data.normal_(0, 0.05)
        # self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s0, t = self.net(in_a).chunk(2, 1)
            log_s = torch.tanh(log_s0)
            s = torch.exp(log_s)
            out_b = (in_b + t) * s

            logdet = torch.sum(log_s.view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s0, t = self.net(out_a).chunk(2, 1)
            log_s = torch.tanh(log_s0)
            s = torch.exp(log_s)
            in_b = out_b / s - t

            logdet = -torch.sum(log_s.view(output.shape[0], -1), 1)

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

            logdet = None

        return torch.cat([out_a, in_b], 1), logdet



class Flow(nn.Module):
	def __init__(self, ndim, affine=True, seqfrac=4):
		super().__init__()

		self.actnorm = ActNorm()
		self.actnorm2 = ActNorm()

		self.coupling = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine)
		self.coupling2 = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine)

		self.ndim = ndim

	def forward(self, input):
		logdet = 0
		out, det1 = self.actnorm(input)
		out, det2 = self.coupling(out)
		out = out[:, np.arange(self.ndim-1, -1, -1)]
		out, det3 = self.actnorm2(out)
		out, det4 = self.coupling2(out)
		out = out[:, np.arange(self.ndim-1, -1, -1)]

		logdet = logdet + det1
		if det2 is not None:
			logdet = logdet + det2
		logdet = logdet + det3
		if det4 is not None:
			logdet = logdet + det4

		return out, logdet

	def reverse(self, output):
		logdet = 0
		input = output[:, np.arange(self.ndim-1, -1, -1)]
		input, det1 = self.coupling2.reverse(input)
		input, det2 = self.actnorm2.reverse(input)
		input = input[:, np.arange(self.ndim-1, -1, -1)]
		input, det3 = self.coupling.reverse(input)
		input, det4 = self.actnorm.reverse(input)


		if det1 is not None:
			logdet = logdet + det1
		logdet = logdet + det2
		if det3 is not None:
			logdet = logdet + det3
		logdet = logdet + det4

		return input, logdet


# class Flow(nn.Module):
#     def __init__(self, ndim, affine=True, seqfrac=4):
#         super().__init__()

#         self.actnorm = ActNorm()
#         self.actnorm2 = ActNorm()

#         self.coupling = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine)
#         self.coupling2 = AffineCoupling(ndim, seqfrac=seqfrac, affine=affine)

#         self.ndim = ndim

#     def forward(self, input):
#         logdet = 0
#         out, det2 = self.coupling(input)
#         out = out[:, np.arange(self.ndim-1, -1, -1)]
#         out, det4 = self.coupling2(out)
#         out = out[:, np.arange(self.ndim-1, -1, -1)]

#         if det2 is not None:
#             logdet = logdet + det2
#         if det4 is not None:
#             logdet = logdet + det4

#         return out, logdet

#     def reverse(self, output):
#         logdet = 0
#         input = output[:, np.arange(self.ndim-1, -1, -1)]
#         input, det1 = self.coupling2.reverse(input)
#         input = input[:, np.arange(self.ndim-1, -1, -1)]
#         input, det3 = self.coupling.reverse(input)


#         if det1 is not None:
#             logdet = logdet + det1
#         if det3 is not None:
#             logdet = logdet + det3

#         return input, logdet


def Order_inverse(order):
	order_inv = []
	for k in range(len(order)):
		for i in range(len(order)):
			if order[i] == k:
				order_inv.append(i)
	return np.array(order_inv)


class RealNVP(nn.Module):
    def __init__(
        self, ndim, n_flow, affine=True, seqfrac=4, permute='random'
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.orders = []
        for i in range(n_flow):
            self.blocks.append(Flow(ndim, affine=affine, seqfrac=seqfrac))
            if permute == 'random':
                self.orders.append(np.random.RandomState(seed=i).permutation(ndim))
            elif permute == 'reverse':
                self.orders.append(np.arange(ndim-1, -1, -1))
            else:
                print('We can only do no permutation, random permutation or reverse permutation in affine coupling layer. Using no permutation by default!')
                self.orders.append(np.arange(ndim))

        self.inverse_orders = []
        for i in range(n_flow):
            self.inverse_orders.append(Order_inverse(self.orders[i]))
    	
    def forward(self, input):
        logdet = 0
        out = input

        for i in range(len(self.blocks)):
            out, det = self.blocks[i](out)
            logdet = logdet + det
            out = out[:, self.orders[i]]

        return out, logdet

    def reverse(self, out):
        logdet = 0
        input = out

        for i in range(len(self.blocks)-1, -1, -1):
            input = input[:, self.inverse_orders[i]]
            input, det = self.blocks[i].reverse(input)
            logdet = logdet + det

        return input, logdet