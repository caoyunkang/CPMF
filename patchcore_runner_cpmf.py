import os
from utils.csv_utils import write_results
from data.mvtec3d_cpmf import get_data_loader
import torch
from tqdm import tqdm
from feature_extractors.cpmf_features import CPMF_Features
import pandas as pd
from data.mvtec3d_cpmf import mvtec3d_classes

class MultiViewPatchCore():
    def __init__(self, backbone_name, dataset_path, n_views, no_fpfh, class_name, root_dir, exp_name, plot_use_rgb, image_size=224):
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.method = CPMF_Features(n_views=n_views, no_fpfh=no_fpfh, backbone_name=backbone_name)
        self.class_name = class_name
        self.image_dir = os.path.join(root_dir, "images", exp_name)
        self.csv_dir = os.path.join(root_dir, "csv")
        self.csv_path = os.path.join(self.csv_dir, f'{exp_name}.csv')
        self.plot_use_rgb = plot_use_rgb

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def fit(self):
        class_name = self.class_name
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, dataset_path=self.dataset_path)
        for sample, _, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):

            self.method.add_sample_to_mem_bank(sample)

        self.method.run_coreset()

    def evaluate(self, draw=False):
        class_name = self.class_name
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, dataset_path=self.dataset_path)
        with torch.no_grad():
            for sample, mask, label in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                self.method.predict(sample, mask, label)

        if draw:
            self.method.draw_anomaly_map(self.image_dir, self.class_name, self.plot_use_rgb)

        self.method.calculate_metrics()
        image_rocauc = round(self.method.image_rocauc, 4)
        pixel_rocauc = round(self.method.pixel_rocauc, 4)
        au_pro = round(self.method.au_pro, 4)
        print(
            f'Class: {class_name}, Image ROCAUC: {image_rocauc:.4f}, Pixel ROCAUC: {pixel_rocauc:.4f}, AU-PRO: {au_pro:.4f}')

        results = dict()
        results['i_roc'] = image_rocauc * 100
        results['p_roc'] = pixel_rocauc * 100
        results['p_pro'] = au_pro * 100

        write_results(results, self.class_name, mvtec3d_classes(), self.csv_path)