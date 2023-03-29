# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.net.voxelize import Voxelization
from lib.dataset.mesh_util import feat_select, read_smpl_constants
from lib.net.MLP import MLP
from lib.net.spatial import SpatialEncoder
from lib.net.HGPIFuCoarseNet import HGPIFuCoarseNet
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import SMPLX
from lib.net.VE import VolumeEncoder
from lib.net.HGFilters import *
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch


class HGPIFuMRNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """

    def __init__(self,
                 cfg,
                 projection_mode="orthogonal",
                 error_term=nn.MSELoss()):

        super(HGPIFuMRNet, self).__init__(projection_mode=projection_mode,
                                        error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        

        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        # TODO: DeleteThis
        # self.smpl_feats = ['sdf', 'cmap', 'norm']
        


        # TODO: add to config
        self.coarse_feature_dim = 128
        self.mlp_dim = [256, 512, 256, 128, 1]
        self.use_filter = self.opt.use_filter_local
        channels_IF = self.mlp_dim

        self.smpl_dim = self.opt.smpl_dim
        #self.smpl_dim = 5
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim

        #TODO: delete this
        self.ignore = ['F_normal_F', 'F_normal_B']


        # t_in_geo = []
        # for item in self.opt.in_geo_local:
        #     if(item[0] not in self.ignore):
        #         t_in_geo.append(item)

        self.in_geo = [item[0] for item in self.opt.in_geo_local]
        self.in_nml = [item[0] for item in self.opt.in_nml_local]
        print(self.in_geo)

        self.use_normal_diff = "C_normal_F" in self.in_geo

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo_local])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml_local])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()
        self.phi = None

        image_lst = [0, 1, 2]
        normal_F_lst = [0, 1, 2] if "image" not in self.in_geo else [3, 4, 5]
        normal_F_lst = [] if "F_normal_F" not in self.in_geo else normal_F_lst

        normal_B_lst = [3, 4, 5] if "image" not in self.in_geo else [6, 7, 8]
        normal_B_lst = [] if "F_normal_B" not in self.in_geo else normal_B_lst


        depth_F_lst = [6] if "image" not in self.in_geo else [9]
        depth_F_lst = [] if "F_depth_F" not in self.in_geo else depth_F_lst
        depth_B_lst = [7] if "image" not in self.in_geo else [10]
        depth_B_lst = [] if "F_depth_B" not in self.in_geo else depth_B_lst
        
        
        # TODO: 
        diff_normal_F_lst = [8, 9, 10] if "image" not in self.in_geo else [11, 12, 13]
        diff_normal_B_lst = [11, 12, 13] if "image" not in self.in_geo else [14, 15, 16]



        # only ICON or ICON-Keypoint use visibility

        if self.prior_type in ["icon", "keypoint"]:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst + depth_F_lst,
                    image_lst + normal_B_lst + depth_B_lst,
                ]
            else:
                if self.use_normal_diff:
                    self.channels_filter = [normal_F_lst + depth_F_lst + diff_normal_F_lst, normal_B_lst + depth_B_lst + diff_normal_B_lst]
                else:
                    self.channels_filter = [normal_F_lst + depth_F_lst, normal_B_lst + depth_B_lst]

        else:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst + normal_B_lst
                ]
            else:
                self.channels_filter = [normal_F_lst + normal_B_lst]

        use_vis = (self.prior_type in ["icon", "keypoint"
                                       ]) and ("vis" in self.smpl_feats)
        if self.prior_type in ["pamir", "pifu"]:
            use_vis = 1

        if self.use_filter:
            channels_IF[0] = (self.hourglass_dim) * (2 - use_vis)
        else:
            channels_IF[0] = len(self.channels_filter[0]) * (2 - use_vis)


        if self.prior_type in ["icon", "keypoint"]:
            channels_IF[0] += self.smpl_dim
        else:
            print(f"don't support {self.prior_type}!")

        channels_IF[0] += self.coarse_feature_dim

        self.base_keys = ["smpl_verts", "smpl_faces"]

        self.icon_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.smpl_feats
        ]
      

        self.if_regressor = MLP(
            filter_channels=channels_IF,
            name="if",
            res_layers=self.opt.res_layers,
            norm=self.opt.norm_mlp,
            last_op=nn.Sigmoid() if not cfg.test_mode else None,
        )

        if self.use_filter:
            self.F_filter = HGFilter(self.opt, self.opt.num_stack,
                                        len(self.channels_filter[0]))
            self.B_filter = HGFilter(self.opt, self.opt.num_stack,
                                        len(self.channels_filter[0]))

        self.sp_encoder = SpatialEncoder()
       

        summary_log = (f"{self.prior_type.upper()}:\n" +
                       f"Image Features used by MLP: {self.in_geo}\n")

        print(colored(summary_log, "yellow"))

        self.netG = HGPIFuCoarseNet(
            cfg,
            projection_mode,
            error_term)
            
        init_net(self)

    def get_normal(self, in_tensor_dict):

        in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo],
                                dim=1)

        return in_filter

    def get_mask(self, in_filter, size=128):

        mask = (F.interpolate(
            in_filter[:, self.channels_filter[0]],
            size=(size, size),
            mode="bilinear",
            align_corners=True,
        ).abs().sum(dim=1, keepdim=True) != 0.0)

        return mask

    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """
        with torch.no_grad():
            global_feature = self.netG.filter(in_tensor_dict, return_inter=False)
        in_filter = self.get_normal(in_tensor_dict)

        features_G = []

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                features_F = self.F_filter(in_filter[:,
                                                     self.channels_filter[0]]
                                           )  # [(B,hg_dim,128,128) * 4]
                features_B = self.B_filter(in_filter[:,
                                                     self.channels_filter[1]]
                                           )  # [(B,hg_dim,128,128) * 4]
            else: 
                features_F = [in_filter[:, self.channels_filter[0]]]
                features_B = [in_filter[:, self.channels_filter[1]]]
            for idx in range(len(features_F)):
                features_G.append(
                    torch.cat([features_F[idx], features_B[idx]], dim=1))

        self.smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        }
        
        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = [features_G[-1]]
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter, global_feature
        else:
            return features_out, global_feature

    def query(self, features, global_features, points, calibs, transforms=None, regressor=None):
        with torch.no_grad():
            sdf = self.netG.query(global_features, points, calibs, transforms, self.netG.if_regressor, update_phi=True)
        xyz = self.projection(points, calibs, transforms)

        (xy, z) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()

        preds_list = []
        vol_feats = features

        if self.prior_type in ["icon", "keypoint"]:

            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # xyz [B, 3, N]  --> points [B, N, 3]

            point_feat_extractor = PointFeat(self.smpl_feat_dict["smpl_verts"],
                                             self.smpl_feat_dict["smpl_faces"])

            point_feat_out = point_feat_extractor.query(
                xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict)

            feat_lst = [
                point_feat_out[key] for key in self.smpl_feats
                if key in point_feat_out.keys()
            ]
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)

            if self.prior_type == "keypoint":
                kpt_feat = self.sp_encoder.forward(
                    cxyz=xyz.permute(0, 2, 1).contiguous(),
                    kptxyz=self.smpl_feat_dict["smpl_joint"],
                )

        
        for im_feat, vol_feat in zip(features, vol_feats):

            # normal feature choice by smpl_vis

            if self.prior_type == "icon":
                if "vis" in self.smpl_feats:
                    point_local_feat = feat_select(self.index(im_feat, xy),
                                                   smpl_feat[:, [-1], :])
                    point_feat_list = [point_local_feat, smpl_feat[:, :-1, :], self.netG.phi]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    # point_feat_list = [point_local_feat, smpl_feat[:, :, :], sdf[-1], self.netG.phi]
                    point_feat_list = [point_local_feat, smpl_feat[:, :, :], self.netG.phi]

            #print(len(sdf))

            point_feat = torch.cat(point_feat_list, 1)

            # out of image plane is always set to 0
            preds = regressor(point_feat, returnPhi=False)
            #if update_phi:
                #self.phi = phi
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    def get_error(self, preds_if_list, labels):
        """calcaulate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0

        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            error_if += self.error_term(pred_if, labels)

        error_if /= len(preds_if_list)

        return error_if

    def forward(self, in_tensor_dict):
        """
        sample_tensor [B, 3, N]
        calib_tensor [B, 4, 4]
        label_tensor [B, 1, N]
        smpl_feat_tensor [B, 59, N]
        """

        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]

       

        in_feat, global_feature = self.filter(in_tensor_dict)

        preds_if_list = self.query(in_feat,
                                   global_feature,
                                   sample_tensor,
                                   calib_tensor,
                                   regressor=self.if_regressor)

        error = self.get_error(preds_if_list, label_tensor)

        return preds_if_list[-1], error
