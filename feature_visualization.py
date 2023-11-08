import argparse
from patchcore_runner_cpmf import MultiViewPatchCore

import os

import loguru
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
from utils.visz_utils import visualize_feature
from matplotlib import pyplot as plt

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


class MultiViewMVTec3DSingleImage(Dataset):

    def __init__(self, split, class_name, defect_type, indx, img_size, dataset_path):
        self.IMAGENET_MEAN = IMAGENET_MEAN
        self.IMAGENET_STD = IMAGENET_STD
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split, defect_type)
        self.defect_type = defect_type
        self.indx = indx
        self.rgb_path = [os.path.join(self.img_path, 'rgb', f"{indx:03d}.png") for indx in self.indx]
        self.gt_path = [os.path.join(self.img_path, 'gt', f"{indx:03d}.png") for indx in self.indx]
        self.xyz_root = [s.replace('rgb', 'xyz')[:-4] for s in self.rgb_path]

        loguru.logger.info(f'img path: {self.img_path}')
        loguru.logger.info(f'rgb path: {self.rgb_path}')
        loguru.logger.info(f'gt path: {self.gt_path}')
        loguru.logger.info(f'xyz root: {self.xyz_root}')

        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

    def __len__(self):
        return len(self.indx)

    def __getitem__(self, idx):
        rgb_path = self.rgb_path[idx]
        xyz_root = self.xyz_root[idx]
        gt = self.gt_path[idx]

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

        return (img, resized_organized_pc, features, view_images, view_positions), gt[:1], 1


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
def get_data_loader(split, class_name, defect_type, indx, img_size, dataset_path, batch_size=1):
    dataset = MultiViewMVTec3DSingleImage(split=split, class_name=class_name, defect_type=defect_type, indx=indx, img_size=img_size,
                                          dataset_path=dataset_path)
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

def concat_features(normal_features, abnormal_features):
    features = np.concatenate([normal_features, abnormal_features], axis=0)
    labels = np.concatenate([[0]*normal_features.shape[0], [1]*abnormal_features.shape[0]], axis=0)
    return features, labels

def run_3d_ads(args):
    cls = args.category
    backbone_name = args.backbone

    print('=========================================')
    kwargs = vars(args)
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print('=========================================')

    print(f"\n {args.exp_name} \n")
    print(f"\nRunning on class {cls}\n")
    patchcore = MultiViewPatchCore(backbone_name=backbone_name, dataset_path=args.data_path, n_views=args.n_views, no_fpfh=args.no_fpfh,
                                   class_name=cls, root_dir=args.root_dir, exp_name=args.exp_name, plot_use_rgb=False)

    # class_name = 'cookie'
    # defect_type = 'crack'
    # image_indx = [0, 3, 5, 6, 7, 8, 1]

    # class_name = 'cable_gland'
    # defect_type = 'bent'
    # image_indx = [0, 3, 5, 6, 7, 8, 1]

    class_name = 'carrot'
    defect_type = 'combined'
    image_indx = [0, 3, 5, 6, 7, 8, 1, 20]

    dataloader = get_data_loader('test', class_name, defect_type, image_indx, 224, 'F:\\VIDF\\multi_view_uniform_mvtec_3d_anomaly_detection')

    normal_vidf_2d_list = []
    abnormal_vidf_2d_list = []

    normal_vidf_3d_list = []
    abnormal_vidf_3d_list = []

    normal_vidf_2d_3d_list = []
    abnormal_vidf_2d_3d_list = []

    for sample, mask, label in dataloader:
        normal_vidf_2d, abnormal_vidf_2d, normal_vidf_3d, abnormal_vidf_3d, normal_vidf_2d_3d, abnormal_vidf_2d_3d = patchcore.method.calculate_normal_abnormal_features(sample ,mask ,label)

        normal_vidf_2d_list.append(normal_vidf_2d.cpu().numpy())
        abnormal_vidf_2d_list.append(abnormal_vidf_2d.cpu().numpy())
        normal_vidf_3d_list.append(normal_vidf_3d.cpu().numpy())
        abnormal_vidf_3d_list.append(abnormal_vidf_3d.cpu().numpy())
        normal_vidf_2d_3d_list.append(normal_vidf_2d_3d.cpu().numpy())
        abnormal_vidf_2d_3d_list.append(abnormal_vidf_2d_3d.cpu().numpy())


    normal_vidf_2d = np.concatenate(normal_vidf_2d_list, axis=0)
    abnormal_vidf_2d = np.concatenate(abnormal_vidf_2d_list, axis=0)
    normal_vidf_3d = np.concatenate(normal_vidf_3d_list, axis=0)
    abnormal_vidf_3d = np.concatenate(abnormal_vidf_3d_list, axis=0)
    normal_vidf_2d_3d = np.concatenate(normal_vidf_2d_3d_list, axis=0)
    abnormal_vidf_2d_3d = np.concatenate(abnormal_vidf_2d_3d_list, axis=0)

    normal_number = min(normal_vidf_2d.shape[0], 2000)
    abnormal_number = min(abnormal_vidf_2d.shape[0], 2000)

    normal_indx_list = np.random.randint(0 ,normal_vidf_2d.shape[0], normal_number)
    abnormal_indx_list = np.random.randint(0, abnormal_vidf_2d.shape[0], abnormal_number)

    normal_vidf_2d = normal_vidf_2d[normal_indx_list, :]
    normal_vidf_3d = normal_vidf_3d[normal_indx_list, :]
    normal_vidf_2d_3d = normal_vidf_2d_3d[normal_indx_list, :]

    abnormal_vidf_2d = abnormal_vidf_2d[abnormal_indx_list, :]
    abnormal_vidf_3d = abnormal_vidf_3d[abnormal_indx_list, :]
    abnormal_vidf_2d_3d = abnormal_vidf_2d_3d[abnormal_indx_list, :]


    save_root = 'E:\\Paper\\Paper-3DAD\\images'

    normal_vidf = [normal_vidf_2d, normal_vidf_3d, normal_vidf_2d_3d]
    abnormal_vidf = [abnormal_vidf_2d, abnormal_vidf_3d, abnormal_vidf_2d_3d]
    tailfix = ['2d','3d','2d_3d']

    for n_f, a_f, prefix in zip(normal_vidf, abnormal_vidf, tailfix):
        features_2d, labels_2d = concat_features(n_f, a_f)
        visualize_feature(features_2d, labels_2d, ['normal','abnormal'], n_components=2, method='TSNE')

        save_path = os.path.join(save_root, f'{class_name}-{defect_type}-{prefix}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-path', type=str, default='../datasets/multi_view_uniform_mvtec_3d_anomaly_detection')
    parser.add_argument('--n-views', type=int, default=27)
    parser.add_argument('--no-fpfh', type=str2bool, default=False)
    parser.add_argument('--use-rgb', type=str2bool, default=False)
    parser.add_argument('--exp-name', type=str, default='uniform-1')
    parser.add_argument('--category', type=str, default='bagel')
    parser.add_argument('--root-dir', type=str, default='./results')
    parser.add_argument('--backbone', type=str, default='resnet18')


    args = parser.parse_args()

    run_3d_ads(args)
