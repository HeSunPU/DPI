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

# alpha-Deep Probabilistic Imaging (alpha-DPI)
![overview image](https://github.com/HeSunPU/DPI/blob/main/planet_astrometry_diagrm.png)
alpha-Deep Probabilistic Inference (alpha-DPI): efficient uncertainty quantification from exoplanet astrometry to black hole feature extraction, [arXiv](https://arxiv.org/abs/2201.08506)

## Run Examples
1. The alpha-DPI radio interferometric example can be trained using ```DPItorch/DPIx_interferometry.py```

    ```python DPIx_interferometry.py --n_gaussian 2 --divergence_type alpha --alpha_divergence 0.95 --n_epoch 20000 --lr 1e-4 --fov 160 --save_path ./checkpoint/interferometry_m87_mcfe/synthetic/crescentfloornuissance2/alpha095closure --obspath ../dataset/interferometry_m87/synthetic_crescentfloorgaussian2/obs_mring_synthdata_allnoise_scanavg_sysnoise2.uvfits```
  
2. The alpha-DPI planet direct imaging orbit fitting example can be trained using ```DPItorch/DPIx_orbit.py```
 
    ```python DPIx_orbit.py --divergence_type alpha --alpha_divergence 0.6 --coordinate_type cartesian --save_path ./checkpoint/orbit_beta_pic_b/cartesian/alpha06```
    
## Citation
```
@article{sun2022alpha,
  title={alpha-Deep Probabilistic Inference (alpha-DPI): efficient uncertainty quantification from exoplanet astrometry to black hole feature extraction},
  author={Sun, He and Bouman, Katherine L and Tiede, Paul and Wang, Jason J and Blunt, Sarah and Mawet, Dimitri},
  journal={arXiv preprint arXiv:2201.08506},
  year={2022}
}

```
