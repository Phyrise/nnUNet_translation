# nnUNet_translation 
For further information, please contact me by e-mail : arthur.longuefosse [at] gmail.com 

Please cite our workshop paper when using nnU-Net_translation :

    Longuefosse, A., Bot, E. L., De Senneville, B. D., Giraud, R., Mansencal, B., Coupé, P., ... & Baldacci, F. (2024, October). 
    Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting. In International Workshop on Simulation and Synthesis in Medical Imaging (pp. 24-33). Cham: Springer Nature Switzerland.

Along with the original nnUNet paper :

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## Tested use cases : 
- Medical cross-modality translation : MR to CT translation
- Medical image inpainting : Inpainting of brain lesions in MR
    
## How to use it : 
```bash
# Please use a dedicated environement to avoid conflit with original nnUNet implementation
git clone https://github.com/Phyrise/nnUNet_translation 
cd nnUNet_translation
pip install -e .
```
The `pip install` command should install the modified [batchgenerators](https://github.com/Phyrise/batchgenerators_translation) and [dynamic-network-architectures](https://github.com/Phyrise/dynamic-network-architectures_translation) repos.

### Please check the files in notebooks/ for the preprocessing steps

Then, export variables :
```bash
export nnUNet_raw="/data/alonguefosse/nnUNet/raw"
export nnUNet_preprocessed="/data/alonguefosse/nnUNet/preprocessed"
export nnUNet_results="/data/alonguefosse/nnUNet/results"
```

now you can train using : 
```bash
nnUNetv2_train DatasetY 3d_fullres 0 -tr nnUNetTrainerMRCT_mae    
```
Several trainers are available :
- L1 loss ([MRCT_mae](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mae.py))
- L1 loss with trilinear interpolation in decoder ([MRCT_mae_trilinear](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mae_trilinear.py)).  Useful to remove checkerboard artifacts. 
- L2 loss ([MRCT_mse](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mse.py))
- Anatomical Feature-Prioritized loss ([MRCT_AFP](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_AFP.py)). Useful to compare features from a pre-trained segmentation network.
Have a look at the [AFP implementation](https://github.com/Phyrise/nnUNet_translation/blob/master/nnunetv2/training/loss/AFP.py)

inference :
```bash
 nnUNetv2_predict -d DatasetY -i INPUT -o OUTPUT -c 3d_fullres -p nnUNetPlans -tr nnUNetTrainerMRCT_mae -f FOLD [optional : -chk checkpoint_best.pth -step_size 0.5 --rec (mean,median)]
```


- A smaller step_size (default: 0.5) at inference can reduce some artifacts on images.
- --rec allows to choose between mean and median reconstruction for overlapping patches 

## TODO : 
- add arguments to control :
    - output channel size (for now : 1)
