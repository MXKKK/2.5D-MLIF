from lib.net import DepthNet
from lib.common.train_util import *
from lib.common.render import Render
import logging
import torch
import numpy as np
from torch import nn
from skimage.transform import resize
import pytorch_lightning as pl
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

logging.getLogger("lightning").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings("ignore")


class Depth(pl.LightningModule):

    def __init__(self, cfg):
        super(Depth, self).__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.lr_N = self.cfg.lr_N
        self.use_vgg = False

        self.schedulers = []

        self.netG = DepthNet(self.cfg, error_term=nn.SmoothL1Loss())



        self.in_nml = [item[0] for item in cfg.net.in_nml]

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

        optim_params_N_F = [{
            "params": self.netG.netF.parameters(),
            "lr": self.lr_N
        }]
        optim_params_N_B = [{
            "params": self.netG.netB.parameters(),
            "lr": self.lr_N
        }]

        optimizer_N_F = torch.optim.Adam(optim_params_N_F,
                                         lr=self.lr_N,
                                         weight_decay=weight_decay)

        optimizer_N_B = torch.optim.Adam(optim_params_N_B,
                                         lr=self.lr_N,
                                         weight_decay=weight_decay)

        scheduler_N_F = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_F, milestones=self.cfg.schedule, gamma=self.cfg.gamma)

        scheduler_N_B = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_N_B, milestones=self.cfg.schedule, gamma=self.cfg.gamma)

        self.schedulers = [scheduler_N_F, scheduler_N_B]
        optims = [optimizer_N_F, optimizer_N_B]

        return optims, self.schedulers

    def render_func(self, render_tensor):
        
        
        height = render_tensor["image"].shape[2]
        result_list = []

        for name in render_tensor.keys():
            if(render_tensor[name].shape[1] == 1):
                render_tensor[name] = torch.repeat_interleave(render_tensor[name], 3, dim=1)
            result_list.append(
                resize(
                    ((render_tensor[name].cpu().numpy()[0] + 1.0) /
                     2.0).transpose(1, 2, 0),
                    (height, height),
                    anti_aliasing=True,
                ))
            
        result_array = np.concatenate(result_list, axis=1)

        return result_array

    def training_step(self, batch, batch_idx, optimizer_idx):

        export_cfg(self.logger, self.cfg)

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]

        FB_tensor = {
            "depth_F": batch["depth_F"],
            "depth_B": batch["depth_B"]
        }

        self.netG.train()

        preds_F, preds_B = self.netG(in_tensor)
        error_NF, error_NB = self.netG.get_norm_error(preds_F, preds_B,
                                                      FB_tensor, self.use_vgg)


        (opt_nf, opt_nb) = self.optimizers()

        opt_nf.zero_grad()
        opt_nb.zero_grad()

        self.manual_backward(error_NF, opt_nf)
        self.manual_backward(error_NB, opt_nb)

        opt_nf.step()
        opt_nb.step()

        if batch_idx > 0 and batch_idx % int(self.cfg.freq_show_train) == 0:

            self.netG.eval()
            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF, "nmlB": nmlB})
                result_array = self.render_func(in_tensor)

                self.logger.experiment.add_image(
                    tag=f"Normal-train/{self.global_step}",
                    img_tensor=result_array.transpose(2, 0, 1),
                    global_step=self.global_step,
                )

        # metrics processing
        metrics_log = {
            "train_loss-NF": error_NF.item(),
            "train_loss-NB": error_NB.item(),
        }

        tf_log = tf_log_convert(metrics_log)
        bar_log = bar_log_convert(metrics_log)

        return {
            "loss": error_NF + error_NB,
            "loss-NF": error_NF,
            "loss-NB": error_NB,
            "log": tf_log,
            "progress_bar": bar_log,
        }

    def training_epoch_end(self, outputs):

        if [] in outputs:
            outputs = outputs[0]

        # metrics processing
        metrics_log = {
            "train_avgloss": batch_mean(outputs, "loss"),
            "train_avgloss-NF": batch_mean(outputs, "loss-NF"),
            "train_avgloss-NB": batch_mean(outputs, "loss-NB"),
        }

        tf_log = tf_log_convert(metrics_log)

        tf_log["lr-NF"] = self.schedulers[0].get_last_lr()[0]
        tf_log["lr-NB"] = self.schedulers[1].get_last_lr()[0]

        return {"log": tf_log}

    def validation_step(self, batch, batch_idx):

        # retrieve the data
        in_tensor = {}
        for name in self.in_nml:
            in_tensor[name] = batch[name]

        FB_tensor = {
            "depth_F": batch["depth_F"],
            "depth_B": batch["depth_B"]
        }

        self.netG.train()

        preds_F, preds_B = self.netG(in_tensor)
        error_NF, error_NB = self.netG.get_norm_error(preds_F, preds_B,
                                                      FB_tensor,self.use_vgg)

        tf =  torch.repeat_interleave(preds_F, 3, dim=1).detach()
        # tf = batch['image']
        F = tf.permute(0,2,3,1).cpu().numpy()
        path = os.path.join('results/depth_new/' + str(batch_idx % 10) + 'FF.jpg')
        plt.imsave(path, (F[0] + 1) / 2)

        if (batch_idx > 0 and batch_idx % int(self.cfg.freq_show_train)
                == 0) or (batch_idx == 0):

            with torch.no_grad():
                nmlF, nmlB = self.netG(in_tensor)
                in_tensor.update({"nmlF": nmlF, "nmlB": nmlB})
                result_array = self.render_func(in_tensor)

                self.logger.experiment.add_image(
                    tag=f"Normal-val/{self.global_step}",
                    img_tensor=result_array.transpose(2, 0, 1),
                    global_step=self.global_step,
                )

        return {
            "val_loss": error_NF + error_NB,
            "val_loss-NF": error_NF,
            "val_loss-NB": error_NB,
        }

    def validation_epoch_end(self, outputs):

        # metrics processing
        metrics_log = {
            "val_avgloss": batch_mean(outputs, "val_loss"),
            "val_avgloss-NF": batch_mean(outputs, "val_loss-NF"),
            "val_avgloss-NB": batch_mean(outputs, "val_loss-NB"),
        }

        tf_log = tf_log_convert(metrics_log)

        return {"log": tf_log}

    def test_single(self, batch):
        in_tensor = {}

        for name in self.in_nml:
            if name in batch:
                in_tensor[name] = batch[name]


        self.netG.eval()

        preds_F, preds_B = self.netG(in_tensor)
        return preds_F, preds_B
    def test_step(self, batch, batch_idx):
               # retrieve the data
        in_tensor = {}

        for name in self.in_nml:
            if name in batch:
                in_tensor[name] = batch[name]


        self.netG.eval()

        preds_F, preds_B = self.netG(in_tensor)

        tf =  torch.repeat_interleave(preds_F, 3, dim=1).detach()
        F = tf.permute(0,2,3,1).cpu().numpy()

        tb =  torch.repeat_interleave(preds_B, 3, dim=1).detach()
        B = tb.permute(0,2,3,1).cpu().numpy()

        #assert(batch['save_path_F'].shape[0] == 1)

        if(not os.path.exists(batch['folder_B'][0])):
            os.mkdir(batch['folder_B'][0])
        if(not os.path.exists(batch['folder_F'][0])):
            os.mkdir(batch['folder_F'][0])

        plt.imsave(batch['save_path_F'][0], (F[0] + 1) / 2)
        plt.imsave(batch['save_path_B'][0], (B[0] + 1) / 2)
