import os

import loguru
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MultiViewMVTec3D(Dataset):

    def __init__(self, split, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = IMAGENET_MEAN
        self.IMAGENET_STD = IMAGENET_STD
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)

        loguru.logger.info(f'img path: {self.img_path}')

        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                rgb_paths.sort()
                xyz_roots = [s.replace('rgb', 'xyz')[:-4] for s in rgb_paths]
                sample_paths = list(zip(rgb_paths, xyz_roots))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                gt_paths.sort()
                xyz_roots = [s.replace('rgb', 'xyz')[:-4] for s in rgb_paths]

                sample_paths = list(zip(rgb_paths, xyz_roots))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        # loguru.logger.info(f'img tot paths: {img_tot_paths}')
        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]

        rgb_path = img_path[0]
        xyz_root = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(img)
        # tiff中，三通道分别代表xyz的数值
        resized_organized_pc, features, view_images, view_positions = self.read_xyz(xyz_root)

        if gt == 0:
            gt = torch.zeros(
                [1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, features, view_images, view_positions), gt[:1], label


    def read_xyz(self, xyz_root):
        # fpfh.npy
        # view_x.png, view_x.npy
        # xyz.tiff
        features = np.load(os.path.join(xyz_root, 'fpfh.npy'))
        organized_pc = read_tiff_organized_pc(os.path.join(xyz_root, 'xyz.tiff'))

        view_image_paths = glob.glob(xyz_root + "/view_*.png")
        view_position_paths = glob.glob(xyz_root + "/view_*.npy")

        view_image_paths.sort()
        view_position_paths.sort()

        view_images = [self.rgb_transform(Image.open(image_path).convert('RGB')) for image_path in view_image_paths]
        view_positions = [np.load(position_path) for position_path in view_position_paths]

        resized_organized_pc = resize_organized_pc(organized_pc)

        return resized_organized_pc, features, view_images, view_positions

# split, class_name, img_size, dataset_path
def get_data_loader(split, class_name, img_size, dataset_path, batch_size=1):
    dataset = MultiViewMVTec3D(split=split, class_name=class_name, img_size=img_size, dataset_path=dataset_path)
    if split in ['train']:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False,
                                 pin_memory=True)
    elif split in ['test']:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False,
                                 pin_memory=True)
    else:
        raise NotImplementedError

    return data_loader


def unorganized_data_to_organized(organized_pc, none_zero_data_list):
    '''

    Args:
        organized_pc:
        none_zero_data_list:

    Returns:

    '''
    if not isinstance(none_zero_data_list, list):
        none_zero_data_list = [none_zero_data_list]

    for idx in range(len(none_zero_data_list)):
        none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().numpy()

    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy() # H W (x,y,z)
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

    full_data_list = []

    for none_zero_data in none_zero_data_list:
        full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
        full_data[nonzero_indices, :] = none_zero_data
        full_data_reshaped = full_data.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], none_zero_data.shape[1]))
        full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
        full_data_list.append(full_data_tensor)

    return full_data_list

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN) * 255.).astype(np.uint8)
    return x
