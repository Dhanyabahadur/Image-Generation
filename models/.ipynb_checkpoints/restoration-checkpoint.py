import torch
import numpy as np
import utils
import os
import torch.nn.functional as F


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
            print('Pre-trained model loaded successfully')
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        print('image folder from restore of DiffusiveRestoration', image_folder)
        with torch.no_grad():
            print('val_loader from restore of DiffusiveRestoration', val_loader)
            for i, (x, y) in enumerate(val_loader):
                print('x and y from val_loader', x, y)
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                print('x_cond.shape:- ', x_cond.shape)
                b, c, h, w = x_cond.shape
                
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                print('x_cond.shape version 2:-', x_cond.shape)
                x_output = self.diffusive_restoration(x_cond)
                print('x_output.shape :- ', x_output.shape)
                x_output = x_output[:, :, :h, :w]
                print('x_output.shape version 2:- ', x_output.shape)
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

