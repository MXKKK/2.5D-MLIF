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

import os
import random
import os.path as osp
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from lib.common.render import Render
from lib.dataset.mesh_util import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class DepthDataset():

    def __init__(self, cfg, split='train'):

        self.split = split
        self.root = cfg.root
        self.bsize = cfg.batch_size
        self.overfit = cfg.overfit

        self.opt = cfg.dataset
        self.datasets = self.opt.types
        self.input_size = self.opt.input_size
        # self.datasets = ['cape']
        self.scales = self.opt.scales
        print('scale')
        print(self.scales)

        # input data types and dimensions
        self.in_nml = [item[0] for item in cfg.net.in_nml]
        self.in_nml_dim = [item[1] for item in cfg.net.in_nml]
        
        self.in_total = self.in_nml + ['depth_F', 'depth_B']
        self.in_total_dim = self.in_nml_dim + [1, 1]
        if(split == 'test'):
            self.in_total = self.in_nml
            self.in_total_dim = self.in_nml_dim
        


        #self.device = torch.device(f"cuda:{cfg.gpus[0]}")
        #self.render = Render(size=1024, dis=100.0, device=self.device)
        #self.smplx = SMPLX()

        # if self.split != 'train':
        #     self.rotations = range(0, 360, 120)
        # else:
        self.rotations = np.arange(0, 360, 360 //
                                   self.opt.rotation_num).astype(np.int)
        # self.rotations = range(0, 360, 120)

        self.datasets_dict = {}

        for dataset_id, dataset in enumerate(self.datasets):

            dataset_dir = osp.join(self.root, dataset)
            smpl_dir = osp.join(dataset_dir, "smpl")


            self.datasets_dict[dataset] = {
                "subjects": np.loadtxt(osp.join(dataset_dir, "all.txt"),
                                       dtype=str),
                "smpl_dir": smpl_dir,
                "scale": self.scales[dataset_id]
            }

        self.subject_list = self.get_subject_list(split)

        # PIL to tensor
        self.image_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # PIL to tensor
        self.mask_to_tensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.0, ), (1.0, ))
        ])

    def render_depth(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_depth_map()

    def load_smpl(self, data_dict):

        smpl_type = "smplx" if ('smplx_path' in data_dict.keys() and
                                os.path.exists(data_dict['smplx_path'])) else "smpl"

        return_dict = data_dict

        if 'smplx_param' in data_dict.keys() and \
            os.path.exists(data_dict['smplx_param']) and \
                sum(self.noise_scale) > 0.0:
            smplx_verts, smplx_dict = self.compute_smpl_verts(data_dict, self.noise_type,
                                                              self.noise_scale)
            smplx_faces = torch.as_tensor(self.smplx.smplx_faces).long()
            #smplx_cmap = torch.as_tensor(np.load(self.smplx.cmap_vert_path)).float()

        else:

            smplx_verts = rescale_smpl(data_dict["smpl_path"], scale=100.0)
            smplx_faces = torch.as_tensor(getattr(self.smplx, f"{smpl_type}_faces")).long()
            #smplx_cmap = self.smplx.cmap_smpl_vids(smpl_type)

        smplx_verts = projection(smplx_verts, data_dict['calib']).float()

        depth, means, stds = self.render_depth(
            (smplx_verts * torch.tensor(np.array([1.0, -1.0, 1.0]))),
            smplx_faces)
        T_depth_F, T_depth_B = depth
        means_F, means_B = means
        stds_F, stds_B = stds
        
        
        #assert(batch['save_path_F'].shape[0] == 1)

        if(not os.path.exists(data_dict['folder_TB'])):
            os.mkdir(data_dict['folder_TB'])
        if(not os.path.exists(data_dict['folder_TF'])):
            os.mkdir(data_dict['folder_TF'])

        del data_dict['folder_TB']
        del data_dict['folder_TF']


        near = -100.0
        far = 100.0
        
        
        # plt.imsave(data_dict['save_path_TF'], ndc_F)
        # plt.imsave(data_dict['save_path_TB'], ndc_B)

        del data_dict['save_path_TF']
        del data_dict['save_path_TB']
        T_depth_F = T_depth_F - 100.0
        T_depth_B = T_depth_B - 100.0
        #T_depth_F = (T_depth_F - 99.0) / 2.0
        #T_depth_F = (T_depth_F - means_F) / stds_F
        #T_depth_F = T_depth_F * 2.0 - 1.0
        #T_depth_B = (T_depth_B - 99.0) / 2.0
        #T_depth_B = (T_depth_B - means_B) / stds_B
        #T_depth_B = T_depth_B * 2.0 - 1.0

        F = T_depth_F[:, :, None].cpu().numpy()
        B = T_depth_B[:, :, None].cpu().numpy()
        F = np.repeat(F, 3, 2)
        B = np.repeat(B, 3, 2)

        ndc_F = F
        ndc_B = B

        #print(np.max(ndc_B))
        #print(np.min(ndc_B))

        T_depth_F = T_depth_F[None, None, :, :]
        T_depth_B = T_depth_B[None, None, :, :]


        return_dict.update({
            'smpl_verts': smplx_verts.cpu(),
            'smpl_faces': smplx_faces.cpu(),
            #'smpl_cmap': smplx_cmap,
        })

        return_dict.update({
            "T_depth_F": T_depth_F.squeeze(0).cpu(),
            "T_depth_B": T_depth_B.squeeze(0).cpu()
        })

        return return_dict

    def load_calib(self, data_dict):
        calib_data = np.loadtxt(data_dict['calib_path'], dtype=float)
        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_mat = torch.from_numpy(calib_mat).float()
        return {'calib': calib_mat}
    def get_subject_list(self, split):

        subject_list = []

        for dataset in self.datasets:

            split_txt = osp.join(self.root, dataset, f'{split}.txt')

            if osp.exists(split_txt):
                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()
            else:
                full_txt = osp.join(self.root, dataset, 'all.txt')
                print(f"split {full_txt} into train/val/test")

                full_lst = np.loadtxt(full_txt, dtype=str)
                full_lst = [dataset + "/" + item for item in full_lst]
                [train_lst, test_lst,
                 val_lst] = np.split(full_lst, [
                     500,
                     500 + 5,
                 ])

                np.savetxt(full_txt.replace("all", "train"),
                           train_lst,
                           fmt="%s")
                np.savetxt(full_txt.replace("all", "test"), test_lst, fmt="%s")
                np.savetxt(full_txt.replace("all", "val"), val_lst, fmt="%s")

                print(f"load from {split_txt}")
                subject_list += np.loadtxt(split_txt, dtype=str).tolist()

        if self.split != 'test':
            subject_list += subject_list[:self.bsize -
                                         len(subject_list) % self.bsize]
            print(colored(f"total: {len(subject_list)}", "yellow"))
            random.shuffle(subject_list)

        # subject_list = ["thuman2/0008"]
        return subject_list

    def __len__(self):
        return len(self.subject_list) * len(self.rotations)

    def __getitem__(self, index):

        # only pick the first data if overfitting
        if self.overfit:
            index = 0

        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        subject = self.subject_list[mid].split("/")[1]
        dataset = self.subject_list[mid].split("/")[0]

        #TODO: do not hardcode 1024 here, use cfg instead
        render_folder = "/".join(
            [dataset + f"_1024_{self.opt.rotation_num}views", subject])

        

        # setup paths
        data_dict = {
            'dataset':
            dataset,
            'subject':
            subject,
            'rotation':
            rotation,
            'scale':
            self.datasets_dict[dataset]["scale"],
            'image_path':
            osp.join(self.root, render_folder, 'render', f'{rotation:03d}.png'),
            'save_path_F':
            osp.join(self.root, render_folder, 'F_depth_F', f'{rotation:03d}.png'),
            'save_path_B':
            osp.join(self.root, render_folder, 'F_depth_B', f'{rotation:03d}.png'),
            'save_path_TF':
            osp.join(self.root, render_folder, 'T_depth_F', f'{rotation:03d}.png'),
            'save_path_TB':
            osp.join(self.root, render_folder, 'T_depth_B', f'{rotation:03d}.png'),
            'folder_TB':
            osp.join(self.root, render_folder, 'T_depth_B'),
            'folder_TF':
            osp.join(self.root, render_folder, 'T_depth_F'),
            'folder_B':
            osp.join(self.root, render_folder, 'F_depth_B'),
            'folder_F':
            osp.join(self.root, render_folder, 'F_depth_F'),
            'calib_path': osp.join(self.root, render_folder, 'calib', f'{rotation:03d}.txt'),
            
        }

        data_dict.update({'smpl_path': osp.join(self.datasets_dict[dataset]["smpl_dir"], f"{subject}.obj")})
        data_dict.update(self.load_calib(data_dict))

        # image/normal/depth loader
        # data_dict = self.load_smpl(data_dict)
        for name, channel in zip(self.in_total, self.in_total_dim):

            if f'{name}_path' not in data_dict.keys():
                
                tmp_path = osp.join(self.root, render_folder, name,
                             f'{rotation:03d}.png')
                
                if(osp.exists(tmp_path)):
                    data_dict.update({
                    f'{name}_path':
                    osp.join(self.root, render_folder, name,
                             f'{rotation:03d}.png')
                    })
                

            # tensor update
            if(name[0] != 'T' ):
                data_dict.update({
                    name:
                    self.imagepath2tensor(data_dict[f'{name}_path'],
                                        channel,
                                        inv=False)
                })

        path_keys = [
            key for key in data_dict.keys() if '_path' in key or '_dir' in key
        ]

        # del data_dict['save_path_TF']
        # del data_dict['save_path_TB']
        #for key in path_keys:
            #del data_dict[key]

        return data_dict

    def imagepath2tensor(self, path, channel=3, inv=False):

        rgba = Image.open(path).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)
        image = (image * mask)[:channel]

        return (image * (0.5 - inv) * 2.0).float()
