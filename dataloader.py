import torch
import os
from torch.utils.data import Dataset
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import json
class GS_dataset:
    def __init__(self,cfg, resolution_scales=[1.0]):
        source_path = cfg["source_path"]
        images = cfg["images"]
        white_background = cfg["white_background"]
        eval = cfg["eval"]
        self.train_cameras = {}
        self.test_cameras = {}


        if os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](source_path, white_background, eval)
        else:
            assert False, "It currently supports only the D-NeRF dataset."

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            cfg)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           cfg)

        self.scene_info = scene_info
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
