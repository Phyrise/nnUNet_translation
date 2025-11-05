# nnUNet_translation 
For further discussion, please contact me by e-mail : arthur.longuefosse [at] gmail.com 

Please cite our workshop paper when using nnU-Net_translation :

    Longuefosse, A., Bot, E. L., De Senneville, B. D., Giraud, R., Mansencal, B., Coupé, P., ... & Baldacci, F. (2024, October). 
    Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting. In International Workshop on Simulation and Synthesis in Medical Imaging (pp. 24-33). Cham: Springer Nature Switzerland.

Along with the original nnUNet paper :

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
    
## Installation
```bash
# Please use a dedicated environment to avoid conflicts with
# the original nnUNet implementation (e.g., venv, conda)
git clone https://github.com/Phyrise/nnUNet_translation 
cd nnUNet_translation
pip install -e .
```
The `pip install` command will also install the modified [batchgenerators](https://github.com/Phyrise/batchgenerators_translation) and [dynamic-network-architectures](https://github.com/Phyrise/dynamic-network-architectures_translation) repos.

## Preprocessing steps
Please check the `notebooks/` files for the preprocessing.

### ISBI 2026 update: we recommend using Residual UNet as the default Plans:
```
nnUNetv2_plan_experiment -d 101 -c 3d_fullres -pl nnUNetPlannerResUNet
```
By default, it integrates trilinear interpolation in the decoder part, instead of transposed convolutions. Standard deconvolution can still be used with ```nnUNetPlannerResUNet_standard```
## Set environment variables
```bash
export nnUNet_raw="/data/nnUNet/raw"
export nnUNet_preprocessed="/data/nnUNet/preprocessed"
export nnUNet_results="/data/nnUNet/results"
```

## Training
```bash
nnUNetv2_train DatasetY 3d_fullres 0 -tr nnUNetTrainerMRCT_mae -pl nnResUNetPlans
```
Several trainers are available :
- L1 loss ([MRCT_mae](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mae.py))
- L2 loss ([MRCT_mse](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mse.py))
- Anatomical Feature-Prioritized loss ([MRCT_AFP](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_AFP.py)). Useful to compare features from a pre-trained segmentation network.
Have a look at the [AFP implementation](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/loss/AFP.py) and our [accepted paper in Physics in Medicine & Biology](https://iopscience.iop.org/article/10.1088/1361-6560/adea07) 

inference :
```bash
nnUNetv2_predict -d DatasetY -i INPUT -o OUTPUT -c 3d_fullres -p nnResUNetPlans -tr nnUNetTrainerMRCT_mae -f FOLD [optional : -chk checkpoint_best.pth -step_size 0.3 --rec (mean,median)]
```

- A smaller step_size (default: 0.5, recommended: 0.3) at inference can reduce some artifacts on images.
- --rec allows selecting the reconstruction method for overlapping patches (```mean```or ```median```).
The ```median``` is still experimental and currently RAM-intensive.

