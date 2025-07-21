import torch, os
from torch import nn, Tensor
import numpy as np
from .unet import PlainConvUNet
from .segairway import SegAirwayModel
import torch.nn.functional as F

"""
from : https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/perceptual.py
return only bottleneck layer ??
"""
class MedicalNetPerceptualSimilarity(nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained by Chen, et al. "Med3D: Transfer
    Learning for 3D Medical Image Analysis". This class uses torch Hub to download the networks from
    "Warvito/MedicalNet-models".

    Args:
        net: {``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"medicalnet_resnet10_23datasets"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(self, net: str = "medicalnet_resnet10_23datasets", verbose: bool = False) -> None:
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load("Warvito/MedicalNet-models", model=net, verbose=verbose)
        self.model = self.model.to(device='cuda', dtype=torch.float16)
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using MedicalNet 3D networks. The input and target tensors are inputted in the
        pre-trained MedicalNet that is used for feature extraction. Then, these extracted features are normalised across
        the channels. Finally, we compute the difference between the input and target features and calculate the mean
        value from the spatial dimensions to obtain the perceptual loss.

        Args:
            input: 3D input tensor with shape BCDHW.
            target: 3D target tensor with shape BCDHW.
        """
        input = medicalnet_intensity_normalisation(input[:,0:1]) #arthur : hardcoded 1 channel
        target = medicalnet_intensity_normalisation(target)

        # input = input.float() #arthur : hardcoded 32 float
        # target = target.float()
        # print(input.dtype)
        # print(target.dtype)

        outs_input = self.model.forward(input)
        outs_target = self.model.forward(target)

        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        # torch.save(feats_input, "feats_input")
        # torch.save(feats_target, "feats_target")

        results = (feats_input - feats_target) ** 2
        results = spatial_average_3d(results.sum(dim=1, keepdim=True), keepdim=True)

        return results.mean()


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)

def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def medicalnet_intensity_normalisation(volume):
    """Based on https://github.com/Tencent/MedicalNet/blob/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b/datasets/brains18.py#L133"""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std


class MedicalNetL1(nn.Module):
    def __init__(self, net: str = "medicalnet_resnet10_23datasets", verbose: bool = False) -> None:
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load("Warvito/MedicalNet-models", model=net, verbose=verbose)
        self.model = self.model.to(device='cuda', dtype=torch.float16)
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = medicalnet_intensity_normalisation(input[:,0:1]) #arthur : hardcoded 1 channel
        target = medicalnet_intensity_normalisation(target)

        outs_input = self.model.forward(input)
        outs_target = self.model.forward(target)

        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        return self.criterion(feats_input, feats_target) #arthur: test with L1 instead of spatial avg


class UNet_layers(nn.Module):
    def __init__(self, net: str = "", layers=[]):
        super().__init__()
        model_params = {
            "TotalSeg_vessels": { #1.5mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_vessels.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
                "num_classes": 3,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_V2": { #patch_size : [128 128 128], 0.6mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "num_classes": 8,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_pelvis_V2": { #0.6mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_pelvis_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 38,
                "model_type": "PlainConvUNet"
            },
            "Imene8": { #96x160x160
                "weights_path": "/data2/alonguefosse/checkpoints/nnUNet_Imene8_best.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "num_classes": 9,
                "model_type": "PlainConvUNet",
            },
            "NaviAirway": {
                "weights_path" : "/data2/alonguefosse/checkpoints/naviairway_semi_supervise.pkl",
                "model_type": "NaviAirway"
            },

            "TotalSeg_HN_V2": { #1*1*3mm
                "weights_path": "/home/phy/Documents/nnUNet/results/Dataset880_TotalSegV2_HN/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 7,
                "model_type": "PlainConvUNet"
            },
        }
        params = model_params[net]
        kernel = params.get("kernels", [[3, 3, 3]] * 6)
        if params["model_type"] == "PlainConvUNet":
            if layers!=[]:
                self.layers = layers
            else:
                self.layers = [0,1,2,3,4,5,6,7,8]
            self.stages = 5
            model = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "NaviAirway":
            if layers!=[]:
                self.layers = layers
            else:
                self.layers = [0,1,2,3,4,5,6]
            self.stages = 4
            model = SegAirwayModel(in_channels=1, out_channels=2)
        
        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda', weights_only = False)
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model.load_state_dict(model_state_dict, strict=False)
        print(f"loaded model : {params['weights_path']}")
        model.eval()
        

        for param in model.parameters(): 
            param.requires_grad = False
        self.model = model    
        self.model = self.model.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 

        self.L1 = nn.L1Loss()
        self.net = net
        self.print_perceptual_layers = False
        self.debug = False
        print("layers : ", self.layers)

    def forward(self, x, y): 
        """
        todo : check if normalization of input tensors is needed
        """
        if self.stages==5:
            padding = (0, 0, 0, 0, 4, 4) #hard coded. TODO: adapt to be multiple of 2^self.stages
            x = F.pad(x, padding, mode='constant', value=0)  
            y = F.pad(y, padding, mode='constant', value=0)  

        emb_x = self.model(x)  
        emb_y = self.model(y)

        sum_loss = 0
        layer_losses = []
        for i in self.layers:
            layer_loss = self.L1(emb_x[i], emb_y[i].detach())
            sum_loss += layer_loss
            layer_losses.append((i, layer_loss.item()))

            if self.print_perceptual_layers:
                print(f"task loss", i, " |", emb_x[i].shape)
                print(layer_loss)

        with open(f'losses_{self.net}.txt', 'a') as file:
            for i, loss in layer_losses:
                file.write(f"Layer {i}: Loss = {loss}\n")
            file.write(f"-------------------\n")

        if self.debug:
            torch.save(x, "embs/x_ep50")
            torch.save(y, "embs/y_ep50")
            for i in range(len(emb_y)):
                torch.save(emb_y[i], f"embs/emb_y_{i}_ep50")
                torch.save(emb_x[i], f"embs/emb_x_{i}_ep50")
            assert(0)
        return sum_loss
    



class L1_UNet_layers(nn.Module):
    def __init__(self, net: str = "", layers=[], mae_weight=10):
        super().__init__()
        model_params = {
            "TotalSeg_vessels": { #1.5mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_vessels.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
                "num_classes": 3,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_V2": { #patch_size : [128 128 128], 0.6mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "num_classes": 8,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_pelvis_V2": { #0.6mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_pelvis_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 38,
                "model_type": "PlainConvUNet"
            },
            "Imene8": { #96x160x160
                "weights_path": "/data2/alonguefosse/checkpoints/nnUNet_Imene8_best.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "num_classes": 9,
                "model_type": "PlainConvUNet",
            },
            "NaviAirway": {
                "weights_path" : "/data2/alonguefosse/checkpoints/naviairway_semi_supervise.pkl",
                "model_type": "NaviAirway"
            },
        
            "TotalSeg_HN_V2": { #1*1*3mm
                "weights_path": "/home/phy/Documents/nnUNet/results/Dataset880_TotalSegV2_HN/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 7,
                "model_type": "PlainConvUNet"
            },
        }
        params = model_params[net]
        kernel = params.get("kernels", [[3, 3, 3]] * 6)
        if params["model_type"] == "PlainConvUNet":
            if layers!=[]:
                self.layers = layers
            else:
                self.layers = [0,1,2,3,4,5,6,7,8]
            self.stages = 5
            model = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "NaviAirway":
            if layers!=[]:
                self.layers = layers
            else:
                self.layers = [0,1,2,3,4,5,6]
            self.stages = 4
            model = SegAirwayModel(in_channels=1, out_channels=2)
        
        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda', weights_only=False)
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model.load_state_dict(model_state_dict, strict=False)
        print(f"loaded model : {params['weights_path']}")
        model.eval()
        

        for param in model.parameters(): 
            param.requires_grad = False
        self.model = model    
        self.model = self.model.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 

        self.L1 = nn.L1Loss()
        self.net = net
        self.print_perceptual_layers = False
        self.debug = False
        self.mae_weight=mae_weight
        print("layers : ", self.layers)
        print("mae_weight :", mae_weight)

    def forward(self, x, y): 
        """
        todo : check if normalization of input tensors is needed
        """

        if self.stages==5:
            padding = (0, 0, 0, 0, 4, 4) #hard coded. TODO: adapt to be multiple of 2^self.stages
            x = F.pad(x, padding, mode='constant', value=0)  
            y = F.pad(y, padding, mode='constant', value=0)  

        emb_x = self.model(x)  
        emb_y = self.model(y)
        

        if self.debug:
            torch.save(x, "embs/x_ep50")
            torch.save(y, "embs/y_ep50")
            for i in range(len(emb_y)):
                torch.save(emb_y[i], f"embs/emb_y_{i}_ep50")
                torch.save(emb_x[i], f"embs/emb_x_{i}_ep50")
            assert(0)

        sum_loss = 0
        layer_losses = []
        for i in self.layers:
            layer_loss = self.L1(emb_x[i], emb_y[i].detach())
            sum_loss += layer_loss
            layer_losses.append((i, layer_loss.item()))

            if self.print_perceptual_layers:
                print(f"task loss", i, " |", emb_x[i].shape)
                print(layer_loss)

        with open(f'losses_{self.net}.txt', 'a') as file:
            for i, loss in layer_losses:
                file.write(f"Layer {i}: Loss = {loss}\n")
            file.write(f"-------------------\n")
            


        mae_loss = self.L1(x.cpu(), y.cpu()) * self.mae_weight
        mae_loss = mae_loss.cuda()

        # mae_loss = 0
        # print("airway : ", sum_loss)
        # print("mae : ", mae_loss)
        # print("--------")
        with open(f'losses_airway_mae.txt', 'a') as file2:
            file2.write(f"airway :  {sum_loss:.3f}")
            file2.write(f" | mae :  {mae_loss:.3f} \n")

        return sum_loss + mae_loss
    




class UNet_layers2(nn.Module):
    def __init__(self, net1 = "", net2 = "", w1 = 1.0, w2 = 1.0):
        super().__init__()
        model_params = {
            "TotalSeg_vessels": { #1.5mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_vessels.pth",
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
                "num_classes": 3,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_V2": { #patch_size : [128 128 128], 0.6mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "num_classes": 8,
                "model_type": "PlainConvUNet"
            },
            "TotalSeg_pelvis_V2": { #0.6mm
                "weights_path": "/data2/alonguefosse/checkpoints/TotalSeg_pelvis_V2.pth", # 5 stage
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "kernels" : [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "num_classes": 38,
                "model_type": "PlainConvUNet"
            },
            "Imene8": { #96x160x160
                "weights_path": "/data2/alonguefosse/checkpoints/nnUNet_Imene8_best.pth", # 5 stage
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
                "num_classes": 9,
                "model_type": "PlainConvUNet",
            },
            "NaviAirway": {
                "weights_path" : "/data2/alonguefosse/checkpoints/naviairway_semi_supervise.pkl",
                "model_type": "NaviAirway"
            },
        }
        params = model_params[net1]
        kernel = params.get("kernels", [[3, 3, 3]] * 6)
        if params["model_type"] == "PlainConvUNet":
            self.layers1 = [0,1,2,3,4,5,6,7,8]
            self.stages1 = 5

            model1 = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                nonlin_kwargs={'inplace': True})
        elif params["model_type"] == "NaviAirway":
            self.layers1 = [0,1,2,3,4,5,6]
            self.stages1 = 4

            model1 = SegAirwayModel(in_channels=1, out_channels=2)

        if not os.path.exists(params["weights_path"]):
            raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
        checkpoint = torch.load(params["weights_path"], map_location='cuda')
        model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
        model1.load_state_dict(model_state_dict, strict=False)
        print(f"loaded model1 : {params['weights_path']}")
        model1.eval()
        for param in model1.parameters(): 
            param.requires_grad = False
        self.model1 = model1   
        self.model1 = self.model1.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 

        if net2!="":
            params = model_params[net2]
            kernel = params.get("kernels", [[3, 3, 3]] * 6)
            if params["model_type"] == "PlainConvUNet":
                self.layers2 = [0,1,2,3,4,5,6,7,8]
                self.stages2 = 5

                model2 = PlainConvUNet(input_channels=1, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], 
                                    conv_op=nn.Conv3d, kernel_sizes=kernel, strides=params["strides"], 
                                    num_classes=params["num_classes"], deep_supervision=False, n_conv_per_stage=[2] * 6, 
                                    n_conv_per_stage_decoder=[2] * 5, conv_bias=True, norm_op=nn.InstanceNorm3d, 
                                    norm_op_kwargs={'eps': 1e-5, 'affine': True}, nonlin=nn.LeakyReLU, 
                                    nonlin_kwargs={'inplace': True})
            elif params["model_type"] == "NaviAirway":
                self.layers2 = [0,1,2,3,4,5,6]
                self.stages2 = 4
                model2 = SegAirwayModel(in_channels=1, out_channels=2)

            if not os.path.exists(params["weights_path"]):
                raise FileNotFoundError(f'Error: Checkpoint not found at {params["weights_path"]}')
            checkpoint = torch.load(params["weights_path"], map_location='cuda')
            model_state_dict = checkpoint.get('state_dict', checkpoint.get('network_weights', checkpoint.get('model_state_dict')))
            model2.load_state_dict(model_state_dict, strict=False)
            print(f"loaded model2 : {params['weights_path']}")
            model2.eval()
            for param in model2.parameters(): 
                param.requires_grad = False
            self.model2 = model2    
            self.model2 = self.model2.to(device='cuda', dtype=torch.float16) #arthur : needed for autocast ? 
        else:
            self.model2 = None

        self.L1 = nn.L1Loss()
        self.net1 = net1
        self.net2 = net2
        self.w1 = w1
        self.w2 = w2
        self.print_perceptual_layers = False
        self.debug = False

    def forward(self, x, y): 
        """
        todo : check if normalization is needed
        since 100% of models are trained on CT = no normalization needed ?
        """
        emb_x1 = self.model1(x[:,0:1])  
        emb_y1 = self.model1(y)

        if self.stages2==5:
            padding = (0, 0, 8, 8, 0, 0) #hard coded for lungs. TODO: adapt to be multiple of 2^self.stages
            x = F.pad(x, padding, mode='constant', value=0)  
            y = F.pad(y, padding, mode='constant', value=0)  

        emb_x2 = self.model2(x[:,0:1])  
        emb_y2 = self.model2(y)

        sum_loss1 = 0
        sum_loss2 = 0
        total_loss = 0
        layer_losses1 = []
        layer_losses2 = []
        for i in self.layers1:
            layer_loss1 = self.L1(emb_x1[i], emb_y1[i].detach())
            sum_loss1 += layer_loss1
            layer_losses1.append((i, layer_loss1.item()))

        for i in self.layers2:
            layer_loss2 = self.L1(emb_x2[i], emb_y2[i].detach())
            sum_loss2 += layer_loss2 
            layer_losses2.append((i, layer_loss2.item()))

        with open(f'losses1_{self.net1}.txt', 'a') as file:
            for i, loss1 in layer_losses1:
                file.write(f"Layer {i}: {self.net1} = {loss1} \n")
            file.write(f"-------------------\n")

        with open(f'losses2_{self.net2}.txt', 'a') as file:
            for i, loss2 in layer_losses2:
                file.write(f"Layer {i}: {self.net2} = {loss2} \n")
            file.write(f"-------------------\n")

        total_loss = sum_loss1 * self.w1 + sum_loss2 * self.w2

        print(self.net1, sum_loss1*self.w1)
        print(self.net2, sum_loss2*self.w2)
        print("----")
        return sum_loss1 + sum_loss2