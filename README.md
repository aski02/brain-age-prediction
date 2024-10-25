# Brain Age Prediction

This repository contains the training pipeline developed for brain age prediction. The goal of this project is to predict brain age from 3D MRI scans using deep learning models such as ResNet-18 and DenseNet-121. 

**Note:** The code in this repository cannot be executed without access to the MRI data, which is not publicly available.

## Repository Structure
```
├── configs/               # Configuration files (YAML)
│   └── config.yaml
├── src/                   # Main source code folder
│   ├── train.py           # Training script
│   ├── data/
│   │   ├── data_loaders/                             # Data loaders for OpenBHB and private dataset
│   │   ├── dataset_loader.py    
│   │   └── data_module_mni_atlas.py                  # Lightning Data Module
│   │   └── data_module_neuromorphometrics_atlas.py   # Lightning Data Module
│   │   └── data_module_whole_brain.py                # Lightning Data Module
│   │   └── data_module_whole_brain_charite.py        # Lightning Data Module
│   └── models/
│       ├── architectures/       # Model architectures
│       └── model.py             # Lightning Module
│       └── model_feature.py     # Lightning Module
│       └── model_multitask.py   # Lightning Module
├── run.sh                 # Script to run the training pipeline
└── README.md
```
