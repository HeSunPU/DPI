# Deep Probabilistic Imaging (DPI)
![overview image](https://github.com/HeSunPU/DPI/blob/main/DPIdiagram.jpg)
Deep Probabilistic Imaging: Uncertainty Quantification and Multi-modal Solution Characterization for Computational Imaging, [AAAI 2021](https://arxiv.org/abs/2010.14462)

## Run Examples
1. The simple 2D example can be run using the ipython notebook ```DPItorch/notebook/DPI toy 2D results.ipynb```

2. The DPI radio interferometric example can be trained using ```DPItorch/DPI_interferometry.py```, and analyzed using ```DPItorch/notebook/DPI interferometry results.ipynb```

    ```python DPI_interferometry.py --lr 1e-4 --clip 1e-3 --n_epoch 30000 --npix 32 --n_flow 16 --logdet 1.0 --save_path ./checkpoint/interferometry --obspath ../dataset/interferometry1/obs.uvfits```
  
3. The DPI MRI example can be trained using ```DPItorch/DPI_interferometry.py```, and analyzed using ```DPItorch/notebook/DPI MRI results.ipynb```

    ```python DPI_MRI.py --lr 1e-5 --clip 1e-3 --n_epoch 100000 --npix 64 --n_flow 16 --ratio 4 --logdet 1.0 --tv 1e3 --save_path ./checkpoint/mri --impath ../dataset/fastmri_sample/mri/knee/scan_0.pkl --maskpath ../dataset/fastmri_sample/mask/mask4.npy --sigma 5e-7```

**Arguments:**

    General:
    * lr (float) - learning rate
    * clip (float) - threshold for gradient clip
    * n_epoch (int) - number of epochs
    * npix (int) - size of reconstruction images (npix * npix)
    * n_flow (int) - number of affine coupling blocks
    * logdet (float) - weight of the entropy loss (larger means more diverse samples)
    * save_path (str) - folder that saves the learned DPI normalizing flow model

    For radio interferometric imaging:
    * obspath (str) - observation data file

    For compressed sensing MRI:
    * impath (str) - fast MRI image for generating MRI measurements
    * maskpath (str) - compressed sensing sampling mask
    * sigma (float) - additive measurement noise
  
## Requirements
General requirements for PyTorch release:
* [pytorch](https://pytorch.org/)
* [torchkbnufft](https://pypi.org/project/torchkbnufft/)

For radio interferometric imaging:
* [eht-imaging](https://pypi.org/project/ehtim/)
* [astropy](https://pypi.org/project/astropy/)
* [pyNFFT](https://pypi.org/project/pyNFFT/)

Please check ```DPI.yml``` for the detailed Anaconda environment information. TensorFlow release is coming soon!

## Citation
```
@inproceedings{sun2021deep,
    author = {He Sun and Katherine L. Bouman},
    title = {Deep Probabilistic Imaging: Uncertainty Quantification and Multi-modal Solution Characterization for Computational Imaging},
    booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
    year = {2021},
}
```
