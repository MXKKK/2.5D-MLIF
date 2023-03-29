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

from lib.common.seg3d_lossless import Seg3dLossless
from lib.dataset.Evaluator import Evaluator
from lib.net import HGPIFuNet
from lib.common.train_util import *
from lib.common.render import Render
from lib.net.MLP import MLP
from lib.dataset.mesh_util import SMPLX, update_mesh_shape_prior_losses, get_visibility
from lib.dataset.PointFeat import PointFeat
import torch
import lib.smplx as smplx
import numpy as np
from torch import nn
from skimage.transform import resize
import pytorch_lightning as pl
from ICONMR import ICONMR

torch.backends.cudnn.benchmark = True


class ICONMR(pl.LightningModule):

    def __init__(self, cfg):
        super(ICONMR, self).__init__()

        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_G = self.cfg.lr_G

        self.use_sdf = cfg.sdf
        self.prior_type = cfg.net.prior_type
        self.mcube_res = cfg.mcube_res
        self.clean_mesh_flag = cfg.clean_mesh

        # self.netG = HGPIFuNet(
        #     self.cfg,
        #     self.cfg.projection_mode,
        #     error_term=nn.SmoothL1Loss() if self.use_sdf else nn.MSELoss(),
        # )

        self.netG = ICON(self.cfg)

        # TODO: change channels here and write to config
        self.RefineMLP = MLP(
            filter_channels=[256],
            name="refineIF",
            res_layers=self.opt.res_layers,
            norm=self.opt.norm_mlp,
            last_op=nn.Sigmoid() if not cfg.test_mode else None,
        )

        self.evaluator = Evaluator(
            device=torch.device(f"cuda:{self.cfg.gpus[0]}"))

        self.resolutions = (np.logspace(
            start=5,
            stop=np.log2(self.mcube_res),
            base=2,
            num=int(np.log2(self.mcube_res) - 4),
            endpoint=True,
        ) + 1.0)
        self.resolutions = self.resolutions.astype(np.int16).tolist()

        self.base_keys = ["smpl_verts", "smpl_faces"]
        self.feat_names = self.cfg.net.smpl_feats

        self.icon_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.feat_names
        ]
        self.keypoint_keys = self.base_keys + [
            f"smpl_{feat_name}" for feat_name in self.feat_names
        ]
        self.pamir_keys = [
            "voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"
        ]
        self.pifu_keys = []

        self.reconEngine = Seg3dLossless(
            query_func=query_func_refine,
            b_min=[[-1.0, 1.0, -1.0]],
            b_max=[[1.0, -1.0, 1.0]],
            resolutions=self.resolutions,
            align_corners=True,
            balance_value=0.50,
            device=torch.device(f"cuda:{self.cfg.test_gpus[0]}"),
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=True,
        )

        self.render = Render(
            size=512, device=torch.device(f"cuda:{self.cfg.test_gpus[0]}"))
        self.smpl_data = SMPLX()

        self.get_smpl_model = lambda smpl_type, gender, age, v_template: smplx.create(
            self.smpl_data.model_dir,
            kid_template_path=osp.join(
                osp.realpath(self.smpl_data.model_dir),
                f"{smpl_type}/{smpl_type}_kid_template.npy",
            ),
            model_type=smpl_type,
            gender=gender,
            age=age,
            v_template=v_template,
            use_face_contour=False,
            ext="pkl",
        )

        self.in_geo = [item[0] for item in cfg.net.in_geo]
        self.in_nml = [item[0] for item in cfg.net.in_nml]
        self.in_geo_dim = [item[1] for item in cfg.net.in_geo]
        self.in_total = self.in_geo + self.in_nml
        self.smpl_dim = cfg.net.smpl_dim

        self.export_dir = None
        self.smpl_feat_dict = None
        self.result_eval = {}

    def eval_coarse(self):
        self.netG.netG.eval()
        self.netG.netG.training = False

    def eval(self):
        self.RefineMLP.eval()

    def train(self):
        self.RefineMLP.train()

    def filter(self, in_tensor_dict, return_inter=False):

        feat_lst = []
        if 'image' in self.in_geo:
            feat_lst.append(in_tensor_dict["image"])
        nmlF = in_tensor_dict["normal_F"]
        nmlB = in_tensor_dict["normal_B"]
        mskF = in_tensor_dict["mask_F"]
        mskB = in_tensor_dict["mask_B"]

        feat_lst.append(nmlF)
        feat_lst.append(mskF)
        feat_lst.append(nmlB)
        feat_lst.append(mskB)
        in_filter = torch.cat(feat_lst, dim=1)

        features_G = []

        
        features_F = [in_filter[:, [0,1,2,3,4,5,6]]]
        features_B = [in_filter[:, [0,1,2,7,8,9,10]]]

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
            return features_out, in_filter
        else:
            return features_out


        
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


    def query(self, features, points, calibs, phi, pred, transforms=None, regressor=None):

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

        
        for im_feat, vol_feat in zip(features, vol_feats):

            # normal feature choice by smpl_vis

            # assert this is always "icon"
            point_local_feat = self.index(im_feat, xy)
            point_feat_list = [point_local_feat, phi, pred]

            point_feat = torch.cat(point_feat_list, 1)

            # out of image plane is always set to 0
            preds = regressor(point_feat)
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    def forward(self, in_tensor_dict, batch, batch_idx):
        preds = self.netG.validation_step(batch, batch_idx, return_pred=True)
        phi = self.netG.phi

        # 需要的是Phi + preds + vis_select(front_normal, back_normal)
        # preds_G, error_G = self.netG(in_tensor_dict)
        in_feat = self.filter(in_tensor_dict)
        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]

        preds_if_list = self.query(in_feat,
                            sample_tensor,
                            calib_tensor,
                            phi,
                            preds,
                            regressor=self.RefineMLP,
                            update_phi=False)

        error = self.get_error(preds_if_list, label_tensor)
        return preds_if_list[-1], error
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if "v_num" in tqdm_dict:
            del tqdm_dict["v_num"]
        return tqdm_dict

    # Training related
    def configure_optimizers(self):

        # set optimizer
        weight_decay = self.cfg.weight_decay
        momentum = self.cfg.momentum

        optim_params_G = [{
            "params": self.RefineMLP.parameters(),
            "lr": self.lr_G
        }]


        if self.cfg.optim == "Adadelta":

            optimizer_G = torch.optim.Adadelta(optim_params_G,
                                               lr=self.lr_G,
                                               weight_decay=weight_decay)

        elif self.cfg.optim == "Adam":

            optimizer_G = torch.optim.Adam(optim_params_G,
                                           lr=self.lr_G,
                                           weight_decay=weight_decay)

        elif self.cfg.optim == "RMSprop":

            optimizer_G = torch.optim.RMSprop(
                optim_params_G,
                lr=self.lr_G,
                weight_decay=weight_decay,
                momentum=momentum,
            )

        else:
            raise NotImplementedError

        # set scheduler
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_G, milestones=self.cfg.schedule, gamma=self.cfg.gamma)

        return [optimizer_G], [scheduler_G]

    def training_step(self, batch, batch_idx):

        if not self.cfg.fast_dev:
            export_cfg(self.logger, self.cfg)


        in_tensor_dict = {
            "sample": batch["samples_geo"].permute(0, 2, 1),
            "calib": batch["calib"],
            "label": batch["labels_geo"].unsqueeze(1),
        }

        for name in self.in_total:
            in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })

        preds_G, error_G = self.forward(in_tensor_dict, batch, batch_idx)
        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            in_tensor_dict["label"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        # metrics processing
        metrics_log = {
            "train_loss": error_G.item(),
            "train_acc": acc.item(),
            "train_iou": iou.item(),
            "train_prec": prec.item(),
            "train_recall": recall.item(),
        }

        tf_log = tf_log_convert(metrics_log)
        bar_log = bar_log_convert(metrics_log)

        if batch_idx % int(self.cfg.freq_show_train) == 0:

            with torch.no_grad():
                self.render_func(in_tensor_dict, dataset="train")

        metrics_return = {
            k.replace("train_", ""): torch.tensor(v)
            for k, v in metrics_log.items()
        }

        metrics_return.update({
            "loss": error_G,
            "log": tf_log,
            "progress_bar": bar_log
        })

        return metrics_return

    def training_epoch_end(self, outputs):

        if [] in outputs:
            outputs = outputs[0]

        # metrics processing
        metrics_log = {
            "train_avgloss": batch_mean(outputs, "loss"),
            "train_avgiou": batch_mean(outputs, "iou"),
            "train_avgprec": batch_mean(outputs, "prec"),
            "train_avgrecall": batch_mean(outputs, "recall"),
            "train_avgacc": batch_mean(outputs, "acc"),
        }

        tf_log = tf_log_convert(metrics_log)

        return {"log": tf_log}

    def validation_step(self, batch, batch_idx):



        in_tensor_dict = {
            "sample": batch["samples_geo"].permute(0, 2, 1),
            "calib": batch["calib"],
            "label": batch["labels_geo"].unsqueeze(1),
        }

        for name in self.in_total:
            in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })
        
        preds_G, error_G = self.forward(in_tensor_dict, batch, batch_idx)

        acc, iou, prec, recall = self.evaluator.calc_acc(
            preds_G.flatten(),
            in_tensor_dict["label"].flatten(),
            0.5,
            use_sdf=self.cfg.sdf,
        )

        if batch_idx % int(self.cfg.freq_show_val) == 0:
            with torch.no_grad():
                self.render_func(in_tensor_dict, dataset="val", idx=batch_idx)

        metrics_return = {
            "val_loss": error_G,
            "val_acc": acc,
            "val_iou": iou,
            "val_prec": prec,
            "val_recall": recall,
        }

        return metrics_return

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val_avgloss": batch_mean(outputs, "val_loss"),
            "val_avgacc": batch_mean(outputs, "val_acc"),
            "val_avgiou": batch_mean(outputs, "val_iou"),
            "val_avgprec": batch_mean(outputs, "val_prec"),
            "val_avgrecall": batch_mean(outputs, "val_recall"),
        }

        tf_log = tf_log_convert(metrics_log)

        return {"log": tf_log}

    def compute_vis_cmap(self, smpl_type, smpl_verts, smpl_faces):

        (xy, z) = torch.as_tensor(smpl_verts).split([2, 1], dim=1)
        smpl_vis = get_visibility(xy, -z, torch.as_tensor(smpl_faces).long())
        smpl_cmap = self.smpl_data.cmap_smpl_vids(smpl_type)

        return {
            "smpl_vis": smpl_vis.unsqueeze(0).to(self.device),
            "smpl_cmap": smpl_cmap.unsqueeze(0).to(self.device),
            "smpl_verts": smpl_verts.unsqueeze(0),
        }


    @torch.enable_grad()
    def optim_cloth(self, verts_pr, faces_pr, inter):

        # convert from GT to SDF
        verts_pr -= (self.resolutions[-1] - 1) / 2.0
        verts_pr /= (self.resolutions[-1] - 1) / 2.0

        losses = {
            "cloth": {
                "weight": 5.0,
                "value": 0.0
            },
            "edge": {
                "weight": 100.0,
                "value": 0.0
            },
            "normal": {
                "weight": 0.2,
                "value": 0.0
            },
            "laplacian": {
                "weight": 100.0,
                "value": 0.0
            },
            "smpl": {
                "weight": 1.0,
                "value": 0.0
            },
            "deform": {
                "weight": 20.0,
                "value": 0.0
            },
        }

        deform_verts = torch.full(verts_pr.shape,
                                  0.0,
                                  device=self.device,
                                  requires_grad=True)
        optimizer_cloth = torch.optim.SGD([deform_verts],
                                          lr=1e-1,
                                          momentum=0.9)
        scheduler_cloth = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cloth,
            mode="min",
            factor=0.1,
            verbose=0,
            min_lr=1e-3,
            patience=5)
        # cloth optimization
        loop_cloth = range(100)

        for i in loop_cloth:

            optimizer_cloth.zero_grad()

            self.render.load_meshes(
                verts_pr.unsqueeze(0).to(self.device),
                faces_pr.unsqueeze(0).to(self.device).long(),
                deform_verts,
            )
            P_normal_F, P_normal_B = self.render.get_rgb_image()

            update_mesh_shape_prior_losses(self.render.mesh, losses)
            diff_F_cloth = torch.abs(P_normal_F[0] - inter[:3])
            diff_B_cloth = torch.abs(P_normal_B[0] - inter[3:])
            losses["cloth"]["value"] = (diff_F_cloth + diff_B_cloth).mean()
            losses["deform"]["value"] = torch.topk(
                torch.abs(deform_verts.flatten()), 30)[0].mean()

            # Weighted sum of the losses
            cloth_loss = torch.tensor(0.0, device=self.device)
            pbar_desc = ""

            for k in losses.keys():
                if k != "smpl":
                    cloth_loss_per_cls = losses[k]["value"] * \
                        losses[k]["weight"]
                    pbar_desc += f"{k}: {cloth_loss_per_cls:.3f} | "
                    cloth_loss += cloth_loss_per_cls

            # loop_cloth.set_description(pbar_desc)
            cloth_loss.backward(retain_graph=True)
            optimizer_cloth.step()
            scheduler_cloth.step(cloth_loss)

        # convert from GT to SDF
        deform_verts = deform_verts.flatten().detach()
        deform_verts[torch.topk(torch.abs(deform_verts),
                                30)[1]] = deform_verts.mean()
        deform_verts = deform_verts.view(-1, 3).cpu()

        verts_pr += deform_verts
        verts_pr *= (self.resolutions[-1] - 1) / 2.0
        verts_pr += (self.resolutions[-1] - 1) / 2.0

        return verts_pr

    def test_step(self, batch, batch_idx):

        self.eval_coarse()
        self.eval()
        in_tensor_dict = {}

        # export paths
        mesh_name = batch["subject"][0]
        mesh_rot = batch["rotation"][0].item()

        self.export_dir = osp.join(self.cfg.results_path, self.cfg.name,
                                   "-".join(self.cfg.dataset.types), mesh_name)

        os.makedirs(self.export_dir, exist_ok=True)

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })

        if "T_normal_F" not in in_tensor_dict.keys(
        ) or "T_normal_B" not in in_tensor_dict.keys():

            # update the new T_normal_F/B
            self.render.load_meshes(
                batch["smpl_verts"] *
                torch.tensor([1.0, -1.0, 1.0]).to(self.device),
                batch["smpl_faces"])
            T_normal_F, T_noraml_B = self.render.get_rgb_image()
            in_tensor_dict.update({
                'T_normal_F': T_normal_F,
                'T_normal_B': T_noraml_B
            })

        with torch.no_grad():

            features, inter = self.filter(in_tensor_dict,
                                               return_inter=True)
            features_coarse = self.netG.netG.filter(in_tensor_dict, return_inter=False)
            sdf = self.reconEngine(opt=self.cfg,
                                   netG=self.netG,
                                   features=features,
                                   features_coarse=features_coarse,
                                   proj_matrix=None)

        def tensor2arr(x):
            return (x[0].permute(1, 2, 0).detach().cpu().numpy() +
                    1.0) * 0.5 * 255.0

        # save inter results
        image = tensor2arr(in_tensor_dict["image"])
        smpl_F = tensor2arr(in_tensor_dict["T_normal_F"])
        smpl_B = tensor2arr(in_tensor_dict["T_normal_B"])
        image_inter = np.concatenate(self.tensor2image(512, inter[0]) +
                                     [smpl_F, smpl_B, image],
                                     axis=1)
        Image.fromarray((image_inter).astype(np.uint8)).save(
            osp.join(self.export_dir, f"{mesh_rot}_inter.png"))

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        verts_gt = batch["verts"][0]
        faces_gt = batch["faces"][0]

        self.result_eval.update({
            "verts_gt": verts_gt,
            "faces_gt": faces_gt,
            "verts_pr": verts_pr,
            "faces_pr": faces_pr,
            "recon_size": (self.resolutions[-1] - 1.0),
            "calib": batch["calib"][0],
        })

        self.evaluator.set_mesh(self.result_eval)
        chamfer, p2s = self.evaluator.calculate_chamfer_p2s(num_samples=1000)
        normal_consist = self.evaluator.calculate_normal_consist(
            osp.join(self.export_dir, f"{mesh_rot}_nc.png"))

        test_log = {"chamfer": chamfer, "p2s": p2s, "NC": normal_consist}

        return test_log

    def test_epoch_end(self, outputs):

        # make_test_gif("/".join(self.export_dir.split("/")[:-2]))

        accu_outputs = accumulate(
            outputs,
            rot_num=3,
            split={
                "cape-easy": (0, 50),
                "cape-hard": (50, 100)
            },
        )

        print(colored(self.cfg.name, "green"))
        print(colored(self.cfg.dataset.noise_scale, "green"))

        self.logger.experiment.add_hparams(
            hparam_dict={
                "lr_G": self.lr_G,
                "bsize": self.batch_size
            },
            metric_dict=accu_outputs,
        )

        np.save(
            osp.join(self.export_dir, "../test_results.npy"),
            accu_outputs,
            allow_pickle=True,
        )

        return accu_outputs

    def tensor2image(self, height, inter):

        all = []
        for dim in self.in_geo_dim:
            img = resize(
                np.tile(
                    ((inter[:dim].cpu().numpy() + 1.0) / 2.0 *
                     255.0).transpose(1, 2, 0),
                    (1, 1, int(3 / dim)),
                ),
                (height, height),
                anti_aliasing=True,
            )

            all.append(img)
            inter = inter[dim:]

        return all

    def render_func(self, in_tensor_dict, dataset="title", idx=0):

        for name in in_tensor_dict.keys():
            if in_tensor_dict[name] is not None:
                in_tensor_dict[name] = in_tensor_dict[name][0:1]

       
        self.eval()

        features, inter = self.filter(in_tensor_dict,
                                    return_inter=True)
        features_coarse = self.netG.netG.filter(in_tensor_dict, return_inter=False)

        sdf = self.reconEngine(opt=self.cfg,
                                netG=self.netG,
                                features=features,
                                features_coarse=features_coarse,
                                proj_matrix=None)


        if sdf is not None:
            render = self.reconEngine.display(sdf)

            image_pred = np.flip(render[:, :, ::-1], axis=0)
            height = image_pred.shape[0]

            image_gt = resize(
                ((in_tensor_dict["image"].cpu().numpy()[0] + 1.0) / 2.0 *
                 255.0).transpose(1, 2, 0),
                (height, height),
                anti_aliasing=True,
            )
            image_inter = self.tensor2image(height, inter[0])
            image = np.concatenate([image_pred, image_gt] + image_inter,
                                   axis=1)

            step_id = self.global_step if dataset == "train" else self.global_step + idx
            self.logger.experiment.add_image(
                tag=f"Occupancy-{dataset}/{step_id}",
                img_tensor=image.transpose(2, 0, 1),
                global_step=step_id,
            )

    def test_single(self, batch):


        self.eval()
        in_tensor_dict = {}

        for name in self.in_total:
            if name in batch.keys():
                in_tensor_dict.update({name: batch[name]})

        in_tensor_dict.update({
            k: batch[k] if k in batch.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        })

        with torch.no_grad():
            self.eval()

            features_coarse = self.netG.netG.filter(in_tensor_dict, return_inter=False)

            # 需要的是Phi + preds + vis_select(front_normal, back_normal)
            # preds_G, error_G = self.netG(in_tensor_dict)
            features, inter = self.filter(in_tensor_dict,
                                  return_inter=True)
            
            # calc features_coarese use ICON
            features_coarse = self.netG.netG.filter(in_tensor_dict, return_inter=False)
            sdf = self.reconEngine(opt=self.cfg,
                                   netG=self,
                                   features=features,
                                   features_coarse = features_coarse,
                                   proj_matrix=None)

        verts_pr, faces_pr = self.reconEngine.export_mesh(sdf)

        if self.clean_mesh_flag:
            verts_pr, faces_pr = clean_mesh(verts_pr, faces_pr)

        verts_pr -= (self.resolutions[-1] - 1) / 2.0
        verts_pr /= (self.resolutions[-1] - 1) / 2.0

        return verts_pr, faces_pr, inter

    
