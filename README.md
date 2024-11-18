# Brain Age Prediction

This repository contains the training pipeline developed for brain age prediction. The goal of this project is to predict brain age from 3D MRI scans using deep learning models such as ResNet-18 and DenseNet-121. 

**Note:** The code in this repository cannot be executed without access to the MRI data, which is not publicly available.

## Repository Structure

Here is the file structure of the project:
```
brain-age-prediction/             # Main project folder
│
├── atlases/                            # Contains atlas files
│   └── labels_Neuromorphometrics.nii   # Neuromorphometrics labels file
│
├── configs/                            # Configuration files for experiments
│   ├── configs_whole_brain.yaml        # Configuration for global brain age
│   └── configs_neuromorphometrics.yaml # Configuration for local brain age
│
├── data/                     # Datahandling
│   ├── data/                 # Contains CSV-Files
│   ├── data_loaders/                   # Dataloaders
│   │   ├── Charite_Dataset.py          # Dataloader for the Charite data
│   │   └── OpenBHB_Dataset.py          # Dataloader for the OpenBHB data
│   ├── brain_age_data_module.py        # Main module for loading the data
│   └── dataset_loader.py     # Dataloader wrapper
│
├── models/                   # Models
│   ├── brain_age_model.py    # Main module for the training procedure
│   ├── densenet_models.py    # Contains DenseNet models
│   └── resnet_models.py      # Contains ResNet models
│
├── preprocessing/            # Preprocessing scripts
│   ├── synthstrip.py         # Script for skull stripping using SynthStrip
│   ├── generate_csv.py       # Script to update the csv file after skull stripping
│   └── quasiraw.py           # Script for performing quasiraw preprocessing
│
├── checkpoints/            # Preprocessing scripts
│   ├── synthstrip.py         # Script for skull stripping using SynthStrip
│   └── quasiraw.py           # Script for performing quasiraw preprocessing
│
├── .gitignore                          # Gitignore file
├── env_training.yml                    # Conda environment file
├── predict_global_age.py               # Global age prediction script
├── predict_local_age.py                # Local age prediction script
├── schedule_global_brain_age.sh        # Script for global brain age training
├── schedule_neuromorphometrics.sh      # Script for local brain age training
└── train.py                            # Training script
```