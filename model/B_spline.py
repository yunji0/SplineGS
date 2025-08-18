import torch
import torch.nn as nn
from copy import deepcopy
import torch

class B_Spline(nn.Module):
    def __init__(self, n_cp, p, clamp_start=False, clamp_end=False):
        assert n_cp > p
        super().__init__()
        self.n_cp = n_cp
        self.p = p
        self.clamp_start = clamp_start
        self.clamp_end = clamp_end
        self.knots = self.create_knots().cuda()

    def create_knots(self):
        #calculate knots from logits
        knots_digits = torch.zeros((self.n_cp + self.p),  dtype=torch.float, device="cuda").contiguous()
        knots_in_between = torch.softmax(knots_digits, dim=0)
        knots_before_pad = knots_in_between.cumsum(dim=0)
        pad_start = knots_before_pad.new_zeros(self.p + 1 if self.clamp_start else 1)
        pad_end = knots_before_pad.new_ones(self.p if self.clamp_end else 0)
        knots = torch.cat((pad_start, knots_before_pad, pad_end), dim=0)
        return knots

    @staticmethod
    def _robust_fraction(num, den):
        return num / (den + 1e-12)

    def forward(self, t, knots =None,sparse_output=False):
        assert 0 <= t <= 1

        # calculate knots
        if knots is not None:
            t_knots = torch.softmax(knots,dim=-1).cumsum(dim=-1)
            t_knots = torch.cat((torch.tensor([0.]).cuda(), t_knots),dim=-1)
        else:
            t_knots =self.knots

        # rescale t if either side is not clamped
        if not (self.clamp_start and self.clamp_end):
            t_start = 0 if self.clamp_start else t_knots[[self.p]]
            t_end = 1 if self.clamp_end else t_knots[[-self.p - 1]]
            t = t_start + t * (t_end - t_start)

        # set base indices and retrieve relevant knots
        idx_cp = (t_knots[1:] <= t).sum(dim=0, keepdim=True).clamp(min=self.p, max=self.n_cp - 1)
        idx_cp = idx_cp + torch.arange(-self.p, self.p + 2, dtype=idx_cp.dtype, device=idx_cp.device)
        # idx_blob = torch.arange(self.n_blob, dtype=idx_cp.dtype, device=idx_cp.device)[:, None]
        t_knots_ret = t_knots[idx_cp]

        # initialize B-spline weights
        _B = t_knots.new_ones(1)

        _dB = t_knots.new_zeros(1)

        for k in range(1, self.p + 1):
            # expand B-spline weights
            _B = nn.functional.pad(_B, (1, 1))
            _dB = nn.functional.pad(_dB, (1, 1))

            # retrieve barriers
            t_i = t_knots_ret[ (self.p - k):(self.p + 1)]
            t_i1 = t_knots_ret[ (self.p - k + 1):(self.p + 2)]
            t_ik = t_knots_ret[ self.p:(self.p + k + 1)]
            t_ik1 = t_knots_ret[ (self.p + 1):(self.p + k + 2)]

            # calculate B-spline weights
            w1 = self._robust_fraction(t - t_i, t_ik - t_i)
            w2 = self._robust_fraction(t_ik1 - t, t_ik1 - t_i1)
            w3 = self._robust_fraction(1, t_ik - t_i)
            w4 = self._robust_fraction(1, t_ik1 - t_i1)
            _B_new = _B[ :-1] * w1 + _B[ 1:] * w2
            _dB_new = _dB[ :-1] * w1 +_B[:-1]*w3 + _dB[1:]*w2 + _B[1:]*w4

            _B = _B_new
            _dB = _dB_new

        _idx_cp = idx_cp[ :(self.p + 1)]
        if sparse_output:
            return _idx_cp, _B, _dB
        else:
            B = _B.new_zeros(self.n_blob, self.n_cp)
            B[ _idx_cp] = _B
            diff_B = _dB.new_zeros(self.n_blob, self.n_cp)
            diff_B[_idx_cp] = _dB
            return B, diff_B*self.p