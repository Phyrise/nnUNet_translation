import torch, os
from torch import nn, Tensor
import numpy as np
from .unet import PlainConvUNet, ResidualEncoderUNet
from .segairway import SegAirwayModel
import torch.nn.functional as F
import math

class AFP_multi_mask3(nn.Module):
    def __init__(self, net1: str = "", net2: str= "", net3= "", layers=[], layers2=[], layers3=[], net1_weight = 1.0, net2_weight = 0.2, net3_weight=0.2, mae_weight=0.0, normalize_before_L1=False):
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
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 118,
                "model_type": "PlainConvUNet"
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
        }
        params = model_params[net1]
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
        print(f"AFP, layers {layers}, loaded {net1} : {params['weights_path']}")
        model.eval()
  
        for param in model.parameters(): 
            param.requires_grad = False
        self.model = model    
        self.model = self.model.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 

        params = model_params[net2]
        kernel = params.get("kernels", [[3, 3, 3]] * 6)
        if params["model_type"] == "PlainConvUNet":
            self.layers2 = layers2 if layers2 else [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.stages = 5
            model2 = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "PlainConvUNet_5":
            self.layers2 = layers2 if layers2 else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model2 = PlainConvUNet(input_channels=1, n_stages=5, features_per_stage=[32, 64, 128, 256, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 5, 
                                n_conv_per_stage_decoder=[2] * 4, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "ResidualEncoderUNet":
            self.layers2 = layers2 if layers2 else [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.stages = 6
            model2 = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=[32, 64, 128, 256, 320, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_blocks_per_stage=[2] * 7, 
                                n_conv_per_stage_decoder=[2] * 6, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "NaviAirway":
            self.layers2 = layers2 if layers2 else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model2 = SegAirwayModel(in_channels=1, out_channels=2)
        
        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda', weights_only = False)
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model2.load_state_dict(model_state_dict, strict=False)
        print(f"AFP, layers2 {layers2}, loaded {net2} : {params['weights_path']}")
        model2.eval()
  
        for param in model2.parameters(): 
            param.requires_grad = False
        self.model2 = model2   
        self.model2 = self.model2.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 

        params = model_params[net3]
        kernel = params.get("kernels", [[3, 3, 3]] * 6)
        if params["model_type"] == "PlainConvUNet":
            self.layers3 = layers3 if layers3 else [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.stages = 5
            model3 = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "PlainConvUNet_5":
            self.layers3 = layers3 if layers3 else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model3 = PlainConvUNet(input_channels=1, n_stages=5, features_per_stage=[32, 64, 128, 256, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 5, 
                                n_conv_per_stage_decoder=[2] * 4, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "ResidualEncoderUNet":
            self.layers3 = layers3 if layers3 else [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.stages = 6
            model3 = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=[32, 64, 128, 256, 320, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_blocks_per_stage=[2] * 7, 
                                n_conv_per_stage_decoder=[2] * 6, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "NaviAirway":
            self.layers3 = layers3 if layers3 else [0, 1, 2, 3, 4, 5, 6]
            self.stages = 4
            model3 = SegAirwayModel(in_channels=1, out_channels=2)
        
        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda', weights_only = False)
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model3.load_state_dict(model_state_dict, strict=False)
        print(f"AFP, layers3 {layers3}, loaded {net3} : {params['weights_path']}")
        model3.eval()
  
        for param in model3.parameters(): 
            param.requires_grad = False
        self.model3 = model3   
        self.model3 = self.model3.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 


        self.L1 = nn.L1Loss()
        self.print_perceptual_layers = False
        self.print_loss = True
        self.debug = False

        self.mae_weight = mae_weight
        self.net1_weight = net1_weight
        self.net2_weight = net2_weight
        self.net3_weight = net3_weight

        self.normalize_before_L1 = normalize_before_L1

    def center_pad_to_multiple_of_2pow(self, x, value=0):
        factor = 2 ** self.stages
        shape = x.shape[-3:]  
        pad = []
        for s in reversed(shape):  # reverse order for F.pad
            new = ((s + factor - 1) // factor) * factor
            total = new - s
            pad.extend([total // 2, total - total // 2])
        return F.pad(x, pad, mode='constant', value=value)
    
    def get_last_layer(self):
        return self.emb_x[-1], self.emb_y[-1]

    def forward(self, x, y, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(x.device).long()

        x = self.center_pad_to_multiple_of_2pow(x)
        y = self.center_pad_to_multiple_of_2pow(y)
        mask = self.center_pad_to_multiple_of_2pow(mask, value=0)

        lung_air = ((mask == 2) | (mask == 3))

        x_lung = x.clone()
        x_lung[lung_air == 0] = -1.0

        y_lung = y.clone()
        y_lung[lung_air == 0] = -1.0

        x_body = x.clone()
        x_body[mask == 0] = -1.0

        y_body = y.clone()
        y_body[mask == 0] = -1.0

        # debug_tensors = {
        # "x.pt": x,
        # "y.pt": y,
        # "x_lung.pt": x_lung,
        # "y_lung.pt": y_lung,
        # "x_body.pt": x_body,
        # "y_body.pt": y_body,
        # }

        # for name, tensor in debug_tensors.items():
        #     print(name)
        #     torch.save(tensor.cpu(), f"{name}")

        # assert(0)

        emb_x1 = self.model(x_lung)
        emb_y1 = self.model(y_lung)

        emb_x2 = self.model2(x_lung)
        emb_y2 = self.model2(y_lung)

        emb_x3 = self.model3(x_body)
        emb_y3 = self.model3(y_body)

        AFP_loss = 0.0
        AFP_loss2 = 0.0
        AFP_loss3 = 0.0

        for i in self.layers:
            AFP_loss += self.L1(emb_x1[i], emb_y1[i].detach())

        for i in self.layers2:
            AFP_loss2 += self.L1(emb_x2[i], emb_y2[i].detach())

        for i in self.layers3:
            AFP_loss3 += self.L1(emb_x3[i], emb_y3[i].detach())

        mae_loss = 0.0
        if self.mae_weight > 0.0:
            mae_loss = self.L1(x, y_body) * self.mae_weight

        AFP_loss *= self.net1_weight
        AFP_loss2 *= self.net2_weight
        AFP_loss3 *= self.net3_weight


        if self.print_loss:
            print(f"AFP_total: {AFP_loss:.5f} | AFP_total2: {AFP_loss2:.5f}  | AFP_total3: {AFP_loss3:.5f} | MAE: {mae_loss:.5f}")

        return AFP_loss + AFP_loss2 + AFP_loss3 + mae_loss

