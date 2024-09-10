import os
import sys
import yaml
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from data.data_module_whole_brain import BrainGenderDataModule
from data.data_module_whole_brain_charite import BrainGenderDataModule as BrainGenderDataModule_charite
from data.data_module_mni_atlas import BrainGenderDataModule as BrainGenderDataModule_mni
from data.data_module_neuromorphometrics_atlas import BrainGenderDataModule as BrainGenderDataModule_neuro
from models.model import BrainGenderModel
from models.model_multitask import BrainGenderModel as MultitaskBrainGenderModel
from models.model_feature import BrainGenderModel as FeatureBrainGenderModel


def load_configs(config_file):
    with open(config_file, 'r') as file:
        configs = yaml.safe_load(file)

    return configs


def run_experiment(config):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    wandb_logger = WandbLogger(
        project=config['project_name'], entity='xxx', name=config['run_name'])
    wandb_logger.experiment.config.update(config)

    if config['split'] == 1:
        data_module = BrainGenderDataModule(config)
    elif config['split'] == 2:
        data_module = BrainGenderDataModule_charite(config)
    elif config['split'] == 3:
        data_module = BrainGenderDataModule_mni(config)
    elif config['split'] == 4:
        data_module = BrainGenderDataModule_neuro(config)

    if config['model_type'] == "normal":
        model = BrainGenderModel(config)
    elif config['model_type'] == "feature" or config['model_type'] == "feature2":
        model = FeatureBrainGenderModel(config)
    elif config['model_type'] == "multitask":
        model = MultitaskBrainGenderModel(config)

    checkpoint_dir = f"checkpoints-thesis/{config['project_name']}/{config['run_name']}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_mae',
        save_top_k=1,
        mode='min',
        dirpath=checkpoint_dir,
        filename='best-checkpoint'
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config['epochs'],
        callbacks=[
            checkpoint_callback,
            pl.callbacks.EarlyStopping(
                monitor='val_mae', min_delta=0.05, patience=config['early_stopping_patience'], mode='min')
        ],
        accelerator="gpu",
        # strategy="ddp",
        devices=[0],
        precision='16-mixed',
        accumulate_grad_batches=config['accumulate_grad_batches']
    )

    trainer.fit(model, datamodule=data_module)

    best_checkpoint_path = checkpoint_callback.best_model_path
    trainer.test(model, datamodule=data_module, ckpt_path=best_checkpoint_path)

    wandb.finish()


if __name__ == "__main__":
    configs_id = int(sys.argv[1])
    configs = load_configs('configs/configs.yaml')

    experiment = configs['experiments'][configs_id]
    run_experiment(experiment)
