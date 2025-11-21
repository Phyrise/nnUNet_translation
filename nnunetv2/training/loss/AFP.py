import torch, os
from torch import nn, Tensor
import numpy as np
from .unet import PlainConvUNet, ResidualEncoderUNet
from .segairway import SegAirwayModel
import torch.nn.functional as F
import math

class AFP(nn.Module):
    def __init__(self, net: str = "", layers=[], mae_weight=0.0, normalize_before_L1=False):
        super().__init__()
        model_params = {
            "TotalSeg_vessels": { #1.5mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_vessels.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
                "num_classes": 3,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_V2": { #patch_size : [128 128 128], 0.6mm, fused 7 labels
                "weights_path": "/export/work/users/arthur/checkpoints/TotalSeg_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "num_classes": 8,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg117": { #patch_size : [128 128 128], 0.6mm, 117 labels kept
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset096_Lungs_117labels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 118,
                "model_type": "PlainConvUNet_5"
            },
            "Imene8": { #96x160x160
                "weights_path": "/export/work/users/arthur/checkpoints/nnUNet_Imene8_best.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "num_classes": 9,
                "model_type": "PlainConvUNet",
            },
            "NaviAirway": {
                "weights_path" : "/export/work/users/arthur/projects/NaviAirway/model_para/checkpoint_semi_supervise_learning.pkl",
                "model_type": "NaviAirway"
            },
            "TotalSeg_AB_7labels": { #1*1*3mm, RIKEN
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset093_SynthRAD2025_AB_7labels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 7,
                "model_type": "PlainConvUNet"
            },
            "Navi_1label": { #O.6*0.6*0.6mm, RIKEN, trained on CHU using Navi labels 
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset095_Lungs_Airways/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 2,
                "model_type": "PlainConvUNet"
            },
            "Navi_2labels": { #O.6*0.6*0.6mm, RIKEN, trained on CHU using Navi labels split (1:trachea, 2:airways)
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset097_Lungs_Airways2/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 3,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_ABHNTH_20labels": { #1*1*3mm, RIKEN
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset091_SynthRAD2025_20labels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 21,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_ABHNTH_117labels": { #1*1*3mm, RIKEN
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset090_SynthRAD2025_117labels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 118,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_AB_7labels_ResEnc": { #1*1*3mm, RIKEN
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset093_SynthRAD2025_AB_7labels/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 7,
                "model_type": "ResidualEncoderUNet"
            },
            "TotalSeg_AB_1x1x1_7labels": { #1*1*3mm, RIKEN
                "weights_path": "/export/work/users/arthur/nnUNet/results/Dataset094_SynthRAD2025_AB_1x1x1_7labels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 7,
                "model_type": "PlainConvUNet"
            },
        }
        params = model_params[net]
        kernel = params.get("kernels", [[3, 3, 3]] * 6)
        if params["model_type"] == "PlainConvUNet":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.stages = 5
            model = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "PlainConvUNet_5":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model = PlainConvUNet(input_channels=1, n_stages=5, features_per_stage=[32, 64, 128, 256, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 5, 
                                n_conv_per_stage_decoder=[2] * 4, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "ResidualEncoderUNet":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.stages = 6
            model = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=[32, 64, 128, 256, 320, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_blocks_per_stage=[2] * 7, 
                                n_conv_per_stage_decoder=[2] * 6, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "NaviAirway":
            self.layers = layers if layers else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model = SegAirwayModel(in_channels=1, out_channels=2)
        
        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda', weights_only = False)
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model.load_state_dict(model_state_dict, strict=False)
        print(f"AFP, layers {layers}, loaded {net} : {params['weights_path']}")
        model.eval()
  
        for param in model.parameters(): 
            param.requires_grad = False
        self.model = model    
        self.model = self.model.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 

        self.L1 = nn.L1Loss()
        self.net = net
        self.print_perceptual_layers = False
        self.print_loss = False
        self.debug = False

        self.mae_weight = mae_weight
        self.normalize_before_L1 = normalize_before_L1

    def center_pad_to_multiple_of_2pow(self, x):
        factor = 2 ** self.stages
        shape = x.shape[-3:]  
        pad = []
        for s in reversed(shape):  # reverse order for F.pad
            new = ((s + factor - 1) // factor) * factor
            total = new - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode='constant', value=0)
    
    def get_last_layer(self):
        return self.emb_x[-1], self.emb_y[-1]

    def forward(self, x, y): 
        """
        todo : check if normalization of input tensors is needed
        """
        x = self.center_pad_to_multiple_of_2pow(x)
        y = self.center_pad_to_multiple_of_2pow(y)

        emb_x = self.model(x)  
        emb_y = self.model(y)

        self.emb_x = emb_x
        self.emb_y = emb_y

        AFP_loss = 0
        layer_losses = []
        for i in self.layers:
            if self.normalize_before_L1:
                emb_x[i] = F.instance_norm(emb_x[i])
                emb_y[i] = F.instance_norm(emb_y[i])
            layer_loss = self.L1(emb_x[i], emb_y[i].detach())
            AFP_loss += layer_loss
            layer_losses.append((i, layer_loss.item()))

            if self.print_perceptual_layers:
                print(f"Layer {i}, {emb_x[i].shape} | L1: {layer_loss.item():.4f}")

        if self.debug:
            with open(f'losses_{self.net}.txt', 'a') as file:
                for i, loss in layer_losses:
                    file.write(f"Layer {i}: Loss = {loss}\n")
                file.write(f"-------------------\n")
            torch.save(x, "embs/x_ep50")
            torch.save(y, "embs/y_ep50")
            for i in range(len(emb_y)):
                torch.save(emb_y[i], f"embs/emb_y_{i}_ep50")
                torch.save(emb_x[i], f"embs/emb_x_{i}_ep50")
            assert(0)

        mae_loss = 0.0
        if self.mae_weight > 0.0:
            mae_loss = self.L1(x, y.float()) * self.mae_weight
            # mae_loss = self.L1(x.cpu(), y.cpu()) * self.mae_weight
            # mae_loss = mae_loss.cuda()
            # with open(f'losses_airway_mae.txt', 'a') as file2:
            #     file2.write(f"airway :  {AFP_loss:.3f}")
            #     file2.write(f" | mae :  {mae_loss:.3f} \n")
        if self.print_loss:
            print(f"AFP_total: {AFP_loss:.5f} | MAE: {mae_loss:.5f}")


        return AFP_loss + mae_loss
