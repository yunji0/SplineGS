import time
import string
import yaml
import os
from gaussian_render import *
import random
import numpy as np
from utils.loss_utils import psnr, ssim, l1_loss

def load_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg
def create_viewpoint_stack(cameras):
    viewpoint_stack = [cameras[x] for x in range(len(cameras))]
    return viewpoint_stack

def training_report(iteration, l1_loss, dataset, gaussians, deform_model, renderFunc, background, cfg):
    if iteration %(cfg["iterations"]//cfg["evaluation_step"]) == 0 or iteration==1:
        torch.cuda.empty_cache()
        if "dyNeRF" in cfg["source_path"]:
            validation_configs = ({'name': 'test', 'cameras': dataset.getTestCameras()},
                                 )

        else:
            validation_configs = ({'name': 'test', 'cameras': dataset.getTestCameras()},{'name': 'train', 'cameras': dataset.getTrainCameras()})

        log_list = []
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # print(idx)

                    image = torch.clamp(renderFunc(viewpoint, gaussians, deform_model, background, cfg)["render"], 0.0, 1.0)

                    if viewpoint.original_image is None:
                        pass
                    else:

                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                log_list.append(psnr_test)
                log_list.append(lpips_test)
                log_list.append(ssim_test)
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        # if cfg["is_log"]:
            # wandb.log(
            #     {"test_psnr": log_list[0], "test_ssim": log_list[2], "test_lpips": log_list[1],"train_psnr": log_list[3],
            #      "n_blob": gaussians.get_xyz.shape[0]})
            # pass
        torch.cuda.empty_cache()



def regulaizer(cfg, gaussians, deform_model, t, pkg):
    is_eval(gaussians,deform_model)
    loss = 0

    # if gaussians.iteration < cfg["densify_until_iter"]:
    loss = loss + opacity_loss(cfg, gaussians)

    return loss

def opacity_loss(cfg, gaussians):
    x = gaussians.get_opacity

    loss = x.mean() *eval(cfg["opacity_lambda"])
    return loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark =False
    torch.backends.cudnn.deterministic = True

def reset_seed():
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def is_train(gaussians,defor_model,itertation):
    gaussians.train=True
    gaussians.iteration = itertation
    defor_model.train=True
    defor_model.iteration=itertation

def is_eval(gaussians,defor_model):
    gaussians.train=False
    defor_model.train=False

def create_experiment_name(n=10):
    rand_str=""
    for i in range(n):
        rand_str += str(random.choice(string.ascii_letters+string.digits))
    return rand_str

def load_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def createDirectory(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        print("Failed to create the directory")

import cv2

def create_chunk_list(A):
    chunck = [A[i:i+300] for i in range(0, len(A), 300)]
    B = []
    for i in range(0,300):
        for j in chunck:
            B.append(random.choice(j))
        random.shuffle(chunck)
    return B
