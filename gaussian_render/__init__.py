import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import pytorch3d.transforms as tr

def render(viewpoint_camera, pc: GaussianModel, deform_model, bg_color: torch.Tensor, cfg, scaling_modifier=1.0, separate_sh=False,
           override_color=None, use_trained_exp=False, t = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,

    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if t is None:
        t = viewpoint_camera.fid
    else:
        t = torch.tensor([t]).cuda()
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation

    if pc.iteration > cfg["warmup"]:
        d_xyz, d_scaling, d_rotation, deform_weight = deform_model.deformation(pc, t)
        means3D = means3D + d_xyz
        rotations = tr.quaternion_multiply(rotations, d_rotation)

    else:
        d_xyz = None
    opacity = pc.get_opacity
    shs = pc.get_features

    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        means2D_densify = screenspace_points_densify,
        shs=shs,
        colors_precomp = None,
        opacities = opacity,
        scales=scales,
        rotations = rotations,
        cov3D_precomp = None)



    out = {
        "render": rendered_image,
        "d_xyz": d_xyz,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth
    }

    return out