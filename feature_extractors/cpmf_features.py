import torch
from feature_extractors.features import Features
from data.mvtec3d_cpmf import unorganized_data_to_organized, denormalization
from torch.nn import functional as F
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
import numpy as np

class CPMF_Features(Features):
    def __init__(self, backbone_name, n_views=1, no_fpfh=False):
        super(CPMF_Features, self).__init__(backbone_name=backbone_name)
        self.n_views = n_views
        self.no_fpfh = no_fpfh

        assert (self.n_views > 0 or not self.no_fpfh)

    # def calculate_single_view_feature(self, image, position):
    #     feature_map = self(image)
    #     position = position.long()
    #
    #     pcd_features = []
    #     for f in feature_map:
    #         f_resize = F.interpolate(f, self.image_size, mode='bilinear')
    #         pcd_features.append(f_resize[:, :, position[0, 1, :], position[0, 0, :]])
    #
    #     pcd_features = torch.cat(pcd_features, 1)
    #
    #     return pcd_features

    def calculate_single_view_feature(self, feature_map, position):
        position = position.long()

        pcd_features = []
        for f in feature_map:
            f_resize = F.interpolate(f, self.image_size, mode='bilinear')
            pcd_features.append(f_resize[:, :, position[0, 1, :], position[0, 0, :]])

        pcd_features = torch.cat(pcd_features, 1)

        return pcd_features

    def calculate_view_invariance_feature(self, sample):
        pcd_feature_list = []

        n_views = min(len(sample[3]), self.n_views)

        feature_map = self(torch.cat(sample[3][:n_views], dim=0))
        for indx, position in enumerate(sample[4][:n_views]):
            pcd_feature_list.append(self.calculate_single_view_feature([f[indx:indx+1, :, :, :] for f in feature_map], position))

        # for view_image, position in zip(sample[3][:n_views], sample[4][:n_views]):
        #     pcd_feature_list.append(self.calculate_single_view_feature(view_image, position))

        pcd_features = torch.cat(pcd_feature_list, 0)
        view_invariant_feature = torch.mean(pcd_features, 0)
        return view_invariant_feature.T

    def add_sample_to_mem_bank(self, sample):
        # sample: img, resized_organized_pc, features, view_images, view_positions
        ############### RGB PATCH ###############
        # perhaps there is no problem? as we follow the origin setting of fpfh
        if self.n_views > 0:
            view_invariant_features = self.calculate_view_invariance_feature(sample)
            view_invariant_features = F.normalize(view_invariant_features, dim=1, p=2)
        ############### END RGB PATCH ###############

        ############### FPFH PATCH ###############
        fpfh_feature_maps = sample[2][0]
        fpfh_feature_maps = F.normalize(fpfh_feature_maps, dim=1, p=2)
        ############### END FPFH PATCH ###############

        if self.n_views > 0 and self.no_fpfh:
            concat_patch = torch.cat([view_invariant_features], dim=1)
        elif self.n_views > 0 and not self.no_fpfh:
            concat_patch = torch.cat([view_invariant_features, fpfh_feature_maps], dim=1)
        else:
            concat_patch = fpfh_feature_maps

        concat_feature_maps = unorganized_data_to_organized(sample[1], [concat_patch])[0]
        concat_feature_maps = self.resize(self.average(concat_feature_maps))
        concat_patch = concat_feature_maps.reshape(concat_feature_maps.shape[1], -1).T

        self.patch_lib.append(concat_patch)

    def predict(self, sample, mask, label):
        ############### RGB PATCH ###############
        if self.n_views > 0:
            view_invariant_features = self.calculate_view_invariance_feature(sample)
            view_invariant_features = F.normalize(view_invariant_features, dim=1, p=2)
        ############### END RGB PATCH ###############

        ############### FPFH PATCH ###############
        fpfh_feature_maps = sample[2][0]
        fpfh_feature_maps = F.normalize(fpfh_feature_maps, dim=1, p=2)
        ############### END FPFH PATCH ###############


        if self.n_views > 0 and self.no_fpfh:
            concat_patch = torch.cat([view_invariant_features], dim=1)
        elif self.n_views > 0 and not self.no_fpfh:
            concat_patch = torch.cat([view_invariant_features, fpfh_feature_maps], dim=1)
        else:
            concat_patch = fpfh_feature_maps

        concat_feature_maps = unorganized_data_to_organized(sample[1], [concat_patch])[0]
        concat_feature_maps = self.resize(self.average(concat_feature_maps))
        concat_patch = concat_feature_maps.reshape(concat_feature_maps.shape[1], -1).T

        self.compute_s_s_map(concat_patch, concat_feature_maps.shape[-2:], mask, label,
                             sample[1][0].permute(1, 2, 0).numpy(), denormalization(sample[0][0].cpu().numpy()))

    def calculate_normal_abnormal_features(self, sample, mask, label):
        ############### RGB PATCH ###############

        view_invariant_features = self.calculate_view_invariance_feature(sample)
        view_invariant_features = F.normalize(view_invariant_features, dim=1, p=2)
        ############### END RGB PATCH ###############

        ############### FPFH PATCH ###############
        fpfh_feature_maps = sample[2][0]
        fpfh_feature_maps = F.normalize(fpfh_feature_maps, dim=1, p=2)
        ############### END FPFH PATCH ###############

        organized_pc_np = sample[1].squeeze().permute(1, 2, 0).numpy()  # H W (x,y,z)
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        mask = mask.squeeze(0).permute(1, 2, 0).reshape(-1, )
        nonzero_mask = mask[nonzero_indices]

        vidf_2d_3d = torch.cat([view_invariant_features, fpfh_feature_maps], dim=1)

        normal_vidf_2d = view_invariant_features[nonzero_mask == 0, :]
        abnormal_vidf_2d = view_invariant_features[nonzero_mask > 0, :]

        normal_vidf_3d = fpfh_feature_maps[nonzero_mask == 0, :]
        abnormal_vidf_3d = fpfh_feature_maps[nonzero_mask > 0, :]

        normal_vidf_2d_3d = vidf_2d_3d[nonzero_mask == 0, :]
        abnormal_vidf_2d_3d = vidf_2d_3d[nonzero_mask > 0, :]

        return normal_vidf_2d, abnormal_vidf_2d, normal_vidf_3d, abnormal_vidf_3d, normal_vidf_2d_3d, abnormal_vidf_2d_3d