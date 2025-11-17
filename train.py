from train_utils import *
from model.deform_model import DeformModel
import gaussian_render
from utils.loss_utils import l1_loss, psnr, ssim
from tqdm import tqdm
from dataloader import GS_dataset

def training(cfg):
    # torch.autograd.set_detect_anomaly(True)
    # set_seed(1004)
    dataset = GS_dataset(cfg)
    cfg["camera_extent"] = dataset.cameras_extent
    viewpoint_stack = create_viewpoint_stack(dataset.getTrainCameras())
    cfg["frame"] = len(viewpoint_stack)
    gaussians = GaussianModel(cfg, sh_degree=3)
    gaussians.training_setup(dataset.scene_info)
    deform_model = DeformModel(cfg)
    deform_model.training_setting()

    ema_loss_for_log = 0.
    n_ =0
    cam_index = [x for x in range(len(viewpoint_stack))]
    bg_color = [1, 1, 1] if cfg["white_background"] else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    progress_bar = tqdm(range(0, cfg["iterations"]), desc="Dynamic_Gaussian_Splatting")

    for iteration in range(1, cfg["iterations"] + 1):
        lr = gaussians.update_learning_rate(iteration)
        is_train(gaussians,deform_model,iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not cam_index:
            cam_index = [x for x in range(len(viewpoint_stack))]
            random.shuffle(cam_index)

        viewpoint_cam = viewpoint_stack[cam_index.pop()]

        render_pkg = gaussian_render.render(viewpoint_cam, gaussians, deform_model, background, cfg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        gt_image = viewpoint_cam.original_image.cuda()


        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg["lambda_dssim"]) * Ll1 + cfg["lambda_dssim"] * (1.0 - ssim(image, gt_image))
        loss = loss #+ regulaizer(cfg,gaussians,deform_model,viewpoint_cam.fid, render_pkg)

        loss.backward()

        with torch.no_grad():
            is_eval(gaussians, deform_model)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "n_blob": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == cfg["iterations"]:
                progress_bar.close()

            early_stop = training_report(iteration, l1_loss, dataset, gaussians,deform_model , render, background, cfg)
            # if early_stop:
            #     break

            # Densification
            if iteration < cfg["densify_until_iter"]:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["d_xyz"])

                if iteration > cfg["densify_from_iter"] and iteration % cfg["densification_interval"] == 0:
                    size_threshold = 20 if iteration > cfg["opacity_reset_interval"] else None
                    gaussians.densify_and_prune(cfg["densify_grad_threshold"], 0.005, dataset.cameras_extent, size_threshold)

                if iteration % cfg["opacity_reset_interval"] == 0 or (
                        cfg["white_background"] and iteration == cfg["densify_from_iter"]):
                    gaussians.reset_opacity()




            # Optimizer step
            if iteration < cfg["iterations"]:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform_model.optimizer.step()
                deform_model.knots.data = torch.clamp(deform_model.knots.data, min = 1e-6)
                deform_model.optimizer.zero_grad(set_to_none=True)
                if iteration <cfg["deform_lr_iter"]:
                    deform_model.scheduler_net.step()

if __name__ == "__main__":
    config = load_config("./config.yaml")
    training(config)

