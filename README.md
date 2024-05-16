# QAERTS: Geometric Transformation Uncertainty for Improving 3D Fetal Brain Pose Prediction from Freehand 2D Ultrasound Videos

## Training
`` python train_end_to_end_vunc8.py [DROPOUT_RATE] [GESTATION_AGE] ``

## Inference
`` python dummy_epi_utils.py [DROPOUT_RATE] [GESTATION_AGE] ``

### Miscellaneous
- Trained weights are provided ``(*.pth)`` in [Google Drive](https://drive.google.com/drive/folders/1fLqaDTRzr85kkWQzumQo_Y0h2AmZq81o?usp=drive_link) for all the models reported in the paper and supplementary, and their defintions can be found in ``epi_models_utils.py``.
- Preprocessing functions for volumes and sampled images can be found and modified for your specific data from ``!dataset_end_to_end_vunc1.py``
- All original geometric transformations are implemented in ``geometry.py``, and adapted in ``epi_models_utils.py``.
- All the necessary/dependent modules are called from these primary scripts. The datasets are *not* provided, but the original sources are mentioned in the paper should you wish to use them.
- Python version used is **3.10.8**. 3.7+ should work.  
- All additonal dependencies for the conda environment used can be found in ``requirements.txt``.




