import os.path
from glob import glob
from stereo_datasets import StereoDataset
from utils import frame_utils

class DrivingStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/home/quant/datasets/drivingstereo/dataset/', image_set='training'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)


        image1_list = sorted(glob(os.path.join(root, image_set+'-left-image/*/*.jpg')))
        image2_list = sorted(glob(os.path.join(root, image_set+'-right-image/*/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, image_set+'-disparity-map/*/*.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
