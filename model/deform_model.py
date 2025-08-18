from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import pytorch3d.transforms as tr
from .B_spline import B_Spline
class simple_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=4):
        super().__init__()
        self.layer1 = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layer,
            },
        )

    def forward(self, x):
        return self.layer1(x)

class DeformModel:
    def __init__(self, cfg):
        self.iteration =- 1
        self.optimizer = None
        self.cfg = cfg
        if hasattr(cfg, 'bound') and cfg.bound is not None:
            self.bound = torch.tensor(cfg.bound, device='cuda')
        else:
            self.bound = torch.tensor([-1., -1., -1., 1., 1., 1.], device='cuda')
        self.spatial_lr_scale = cfg.camera_extent
        self.hidden_dim = cfg.hidden_dim

        self.n_cp = cfg.n_cp
        self.degree = cfg.degree
        self.b_spliner = B_Spline(self.n_cp, self.degree)

        self.deform_HASH = tcnn.Encoding(
                 n_input_dims=3,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": cfg.n_levels,
                    "n_features_per_level": cfg.n_features_per_level,
                    "log2_hashmap_size": cfg.log2_hashmap_size,
                    "base_resolution": cfg.base_resolution,
                    "per_level_scale": cfg.per_level_scale,
                },

        )

        curve = (torch.randn(cfg.n_w,self.n_cp,6)*1e-5).contiguous().cuda()
        self.curve = torch.nn.Parameter(curve.requires_grad_(True))
        knots = torch.ones(cfg.n_w,self.n_cp).contiguous().cuda()
        self.knots = torch.nn.Parameter(knots.requires_grad_(True))
        self.mlp_head = simple_network(self.deform_HASH.n_output_dims,self.hidden_dim, cfg.n_w,cfg.n_layers).cuda()



    def deformation(self, pc, t):

        idx_cp, weights, _ = self.b_spliner(t, sparse_output=True)
        key, _ = self.contract_to_unisphere(pc._xyz.clone().detach(),self.bound)

        deform_feature = self.deform_HASH(key)
        deform_weight = self.mlp_head(deform_feature).float()

        b_weights = weights.unsqueeze(0)
        cp_weights = self.knots[:, idx_cp]
        weight_ = (b_weights*cp_weights)
        weight_ = weight_ / weight_.sum(dim=-1,keepdim=True)

        deform_weight = torch.tanh(deform_weight)

        repre_curve = torch.einsum('wcf,wc-> wf', self.curve[:, idx_cp], weight_)
        deform_ = torch.einsum('nw,wf->nf', deform_weight, repre_curve[...,:6])

        d_xyz = deform_[...,:3]
        d_scaling = None
        d_rotation = tr.axis_angle_to_quaternion(deform_[...,3:6])

        return d_xyz, d_scaling, d_rotation, deform_weight

    def training_setting(self):


        l = [
            {'params': self.deform_HASH.parameters(), 'lr': 0.01*self.cfg.lr_scale * self.spatial_lr_scale, "name": "deform_HASH", "weight_decay": self.cfg.weight_decay},
            {'params': self.mlp_head.parameters(), 'lr': 0.001*self.cfg.lr_scale, "name": "mlp_head", "weight_decay": self.cfg.weight_decay},


            {'params': [self.curve], 'lr': self.cfg.curve_lr, "name": "curve",},
            {'params': [self.knots], 'lr': self.cfg.weight_lr, "name": "knots"},

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01/(self.cfg.warmup/100), total_iters=self.cfg.warmup
                ),

                torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=100,
                    gamma=self.cfg.lr_lambda,
                ),
            ]
        )

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x,mask