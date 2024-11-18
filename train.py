import os
import sys
import yaml
import wandb
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger

from data.brain_age_data_module import BrainAgeDataModule
from models.brain_age_model import BrainAgeModel

# Specify which neuromorphometrics regions should be trained on which GPUs when we utilize multiple GPUs
GPU_REGION_MAPPING = {
    0: [44, 191, 168, 107, 148, 23, 46],
    1: [45, 154, 169, 195, 149, 75, 32],
    2: [39, 155, 132, 52, 61, 112, 161, 47, 204, 171, 125, 185, 119, 117, 165, 31, 49, 64, 63, 30],
    3: [38, 183, 41, 194, 167, 102, 166, 162, 121, 151, 170, 124, 137, 104, 206, 187, 4, 50, 69, 76],
    4: [143, 182, 133, 202, 200, 129, 193, 139, 115, 100, 57, 109, 157, 37, 150, 173, 118, 141, 207, 55],
    5: [142, 177, 40, 152, 51, 153, 145, 144, 138, 146, 196, 120, 163, 36, 179, 174, 181, 184, 11, 186],
    6: [35, 198, 176, 203, 135, 123, 201, 192, 101, 71, 103, 113, 160, 197, 156, 73, 172, 72, 140, 116, 56],
    7: [190, 199, 106, 134, 122, 60, 59, 128, 114, 62, 147, 58, 205, 108, 48, 175, 178, 136, 180, 105, 164]
}


def load_configs(config_file):
    with open(config_file, 'r') as file:
        configs = yaml.safe_load(file)
    return configs


def run_experiment(config, gpu_id):
    seed = config.get('seed', 42)
    pl.seed_everything(seed, workers=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    os.environ["WANDB_MODE"] = config.get('wandb_mode', 'online')

    if config.get('atlas_type') == 'neuromorphometrics' and config.get('crop_type') == 'region':
        regions_ids = GPU_REGION_MAPPING.get(gpu_id, [])
        if not regions_ids:
            print(f"No regions assigned to GPU {gpu_id}")
            return

        for region_id in regions_ids:
            config['region_id'] = region_id
            original_run_name = config['run_name']
            config['run_name'] = f"{original_run_name}_region_{region_id}"

            wandb_logger = WandbLogger(
                project=config['project_name'],
                name=config['run_name'],
                entity=config.get('wandb_entity', 'default'),
                log_model=None
            )
            wandb_logger.experiment.config.update(config)

            data_module = BrainAgeDataModule(config)
            model = BrainAgeModel(config)

            checkpoint_dir = os.path.join(
                'checkpoints', config['project_name'], config['run_name'])
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor='val_mae',
                save_top_k=1,
                mode='min',
                dirpath=checkpoint_dir,
                filename='best-checkpoint'
            )

            callbacks = [checkpoint_callback]

            if config.get('use_early_stopping', False):
                early_stopping_callback = pl.callbacks.EarlyStopping(
                    monitor='val_mae',
                    min_delta=0.05,
                    patience=config.get('early_stopping_patience', 10),
                    mode='min',
                    verbose=True
                )
                callbacks.append(early_stopping_callback)

            trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=config['epochs'],
                callbacks=callbacks,
                accelerator='gpu',
                devices=[0],
                precision='16-mixed',
                accumulate_grad_batches=config.get(
                    'accumulate_grad_batches', 1)
            )

            trainer.fit(model, datamodule=data_module)

            best_checkpoint_path = checkpoint_callback.best_model_path
            if best_checkpoint_path:
                trainer.test(
                    model=model, ckpt_path=best_checkpoint_path, datamodule=data_module)
            else:
                trainer.test(model=model, datamodule=data_module)

            wandb.finish()

            config['run_name'] = original_run_name
    else:
        callbacks = []

        wandb_logger = WandbLogger(
            project=config['project_name'],
            name=config['run_name'],
            entity=config.get('wandb_entity', 'default'),
            log_model=None
        )
        wandb_logger.experiment.config.update(config)

        data_module = BrainAgeDataModule(config)
        model = BrainAgeModel(config)

        checkpoint_dir = os.path.join(
            'checkpoints', config['project_name'], config['run_name'])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_mae',
            save_top_k=1,
            mode='min',
            dirpath=checkpoint_dir,
            filename='best-checkpoint'
        )
        callbacks.append(checkpoint_callback)

        if config.get('use_early_stopping', False):
            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor='val_mae',
                min_delta=0.05,
                patience=config.get('early_stopping_patience', 10),
                mode='min',
                verbose=True
            )
            callbacks.append(early_stopping_callback)

        devices = [gpu_id] if torch.cuda.is_available() else None
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=config['epochs'],
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision='16-mixed' if torch.cuda.is_available() else 32,
            accumulate_grad_batches=config.get('accumulate_grad_batches', 1)
        )

        trainer.fit(model, datamodule=data_module)

        best_checkpoint_path = checkpoint_callback.best_model_path
        if best_checkpoint_path:
            trainer.test(model=model, ckpt_path=best_checkpoint_path,
                         datamodule=data_module)
        else:
            trainer.test(model=model, datamodule=data_module)

        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Brain Age Model')
    parser.add_argument('--config_file', type=str,
                        default='configs/configs_whole_brain.yaml', help='Path to the configuration file')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID for this process (default: 0)')
    args = parser.parse_args()

    configs = load_configs(args.config_file)
    experiment = configs

    try:
        run_experiment(experiment, args.gpu_id)
    except Exception as e:
        print(f"Error during experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
