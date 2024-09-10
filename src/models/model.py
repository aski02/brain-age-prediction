import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np
from monai.transforms import RandRotated, RandAffined, EnsureChannelFirstd
import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef, R2Score
from architectures.densenet3d import SupRegDenseNet as DenseNet
from architectures.resnet3d import SupRegResNet as ResNet
from architectures.resnet3d import SupRegResNetDropout as ResNetDropout
from architectures.densenet3d import SupRegDenseNetDropout as DenseNetDropout
from torchmetrics import PearsonCorrCoef, R2Score, Accuracy, AUROC
from torchmetrics import Precision, Recall, F1Score, Specificity, ConfusionMatrix
import matplotlib.pyplot as plt


class BrainGenderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config['dropout'] == 1:
            if self.config['model'] == "ResNet":
                self.model = ResNetDropout()
            elif self.config['model'] == "DenseNet":
                self.model = DenseNetDropout()
        else:
            if self.config['model'] == "ResNet":
                self.model = ResNet()
            elif self.config['model'] == "DenseNet":
                self.model = DenseNet()

        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.pearson_corr = PearsonCorrCoef(num_outputs=1)
        self.r2_score = R2Score(num_outputs=1)

        max_rad = np.radians(15)
        self.random_rotation = RandRotated(
            keys=["image"],
            range_x=(-max_rad, max_rad),
            range_y=(-max_rad, max_rad),
            range_z=(-max_rad, max_rad),
            prob=0.5,
            mode='bilinear',
        )
        self.random_affine = RandAffined(
            keys=["image"],
            prob=0.5,
            translate_range=(10, 10, 10),
            mode='bilinear',
            padding_mode='zeros',
            cache_grid=True,
            spatial_size=(182, 218, 182)
        )

        self.test_outputs = {}
        self.test_labels = {}
        self.test_genders = {}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config['optimizer'] == 'Adam':
            optimizer = optim.Adam([
                {'params': self.model.parameters()},
            ], lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW([
                {'params': self.model.parameters()},
            ], lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            optimizer = optim.SGD([
                {'params': self.model.parameters()},
            ], lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

        if self.config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=10, threshold=0.05, threshold_mode='abs', verbose=True)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_mae'}

        elif self.config['scheduler'] == 'StepLR':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif self.config['scheduler'] == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=0
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=35,
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        images, ages, genders, scanner = batch['image'], batch['label'].float(
        ), batch['gender'].float(), batch['scanner'].int()

        if self.config['augmentation'] == 1:
            transformed_images = []
            for i in range(images.shape[0]):
                single_image = images[i]
                transformed_image = {'image': single_image}
                transformed_image = self.random_rotation(transformed_image)
                transformed_image = self.random_affine(transformed_image)
                transformed_images.append(transformed_image['image'])
            images = torch.stack(transformed_images, dim=0)
        elif self.config['augmentation'] == 2:
            transformed_images = []
            for i in range(images.shape[0]):
                single_image = images[i]
                transformed_image = {'image': single_image}
                transformed_image = self.random_affine(transformed_image)
                transformed_images.append(transformed_image['image'])
            images = torch.stack(transformed_images, dim=0)
        elif self.config['augmentation'] == 3:
            transformed_images = []
            for i in range(images.shape[0]):
                single_image = images[i]
                transformed_image = {'image': single_image}
                transformed_image = self.random_rotation(transformed_image)
                transformed_images.append(transformed_image['image'])
            images = torch.stack(transformed_images, dim=0)

        outputs, _ = self(images)
        ages = ages.unsqueeze(1)

        mse_loss = self.mse_criterion(outputs, ages)
        mae_loss = self.mae_criterion(outputs, ages)

        if self.config['loss_weighted'] == 1:
            weight = torch.ones_like(ages)
            weight[scanner == 2] = 3
            weight[scanner == 1] = 3
            mse_loss = (mse_loss * weight).mean()
            mae_loss = (mae_loss * weight).mean()
            self.log('train_mse', mse_loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
            self.log('train_mae', mae_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
        elif self.config['loss_weighted'] == 2:
            weight = torch.ones_like(ages)
            weight[scanner == 2] = 10
            weight[scanner == 1] = 10
            mse_loss = (mse_loss * weight).mean()
            mae_loss = (mae_loss * weight).mean()
            self.log('train_mse', mse_loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
            self.log('train_mae', mae_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
        elif self.config['loss_weighted'] == 3:
            weight = torch.ones_like(ages)
            weight[scanner == 2] = 50
            weight[scanner == 1] = 50
            mse_loss = (mse_loss * weight).mean()
            mae_loss = (mae_loss * weight).mean()
            self.log('train_mse', mse_loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
            self.log('train_mae', mae_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
        else:
            self.log('train_mse', mse_loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))
            self.log('train_mae', mae_loss, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=len(images))

        return mae_loss

    def validation_step(self, batch, batch_idx):
        images, ages, genders = batch['image'], batch['label'].float(
        ), batch['gender'].float()
        outputs, _ = self(images)
        ages = ages.unsqueeze(1)
        mse_loss = self.mse_criterion(outputs, ages)
        mae_loss = self.mae_criterion(outputs, ages)
        r_value = self.pearson_corr(outputs, ages)
        r_squared = self.r2_score(outputs, ages)
        self.log('val_mse', mse_loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=len(images))
        self.log('val_mae', mae_loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=len(images))
        self.log('val_r_value', r_value, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=len(images))
        self.log('val_r_squared', r_squared, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=len(images))
        return mae_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_names = ['hc', 'mogad', 'nmosd', 'ms_trio',
                         'ms_prisma', 'ms_berl', 'val', 'val_charite', 'val_openbhb']
        dataset_name = dataset_names[dataloader_idx]

        images, ages, genders = batch['image'], batch['label'].float(
        ), batch['gender'].float()
        outputs, _ = self(images)
        ages = ages.unsqueeze(1)

        if dataset_name not in self.test_outputs:
            self.test_outputs[dataset_name] = []
            self.test_labels[dataset_name] = []
            self.test_genders[dataset_name] = []

        self.test_outputs[dataset_name].append(outputs)
        self.test_labels[dataset_name].append(ages)
        self.test_genders[dataset_name].append(
            genders)

    def on_test_epoch_end(self):
        dataset_names = ['hc', 'mogad', 'nmosd', 'ms_trio',
                         'ms_prisma', 'ms_berl', 'val', 'val_charite', 'val_openbhb']

        for dataset_name in dataset_names:
            if dataset_name in self.test_outputs:
                outputs = torch.cat(self.test_outputs[dataset_name], dim=0)
                ages = torch.cat(self.test_labels[dataset_name], dim=0)
                genders = torch.cat(self.test_genders[dataset_name], dim=0)

                mae_loss = self.mae_criterion(outputs, ages)
                r_value = self.pearson_corr(outputs, ages)

                self.log(f'test_mae_{dataset_name}', mae_loss, on_step=False,
                         on_epoch=True, prog_bar=False, logger=True, batch_size=len(genders))
                self.log(f'test_r_value_{dataset_name}', r_value, on_step=False,
                         on_epoch=True, prog_bar=False, logger=True, batch_size=len(genders))

                true_ages_list = ages.squeeze().cpu().numpy().tolist()
                predicted_ages_list = outputs.squeeze().cpu().numpy().tolist()
                genders_list = genders.squeeze().cpu().numpy(
                ).tolist()

                wandb.log({f'{dataset_name}_true_ages': true_ages_list})
                wandb.log(
                    {f'{dataset_name}_predicted_ages': predicted_ages_list})
                wandb.log({f'{dataset_name}_genders': genders_list})

                plt.figure(figsize=(10, 6))
                plt.scatter(ages.cpu(), outputs.cpu(), alpha=0.5, color='blue')
                plt.xlabel('Age')
                plt.ylabel('Predicted Age')
                plt.title(f'True vs Predicted Ages for {dataset_name.upper()}')
                plt.grid(True)
                plt.xlim(10, 90)
                plt.ylim(10, 90)
                plt.plot([10, 90], [10, 90], color='red', linestyle='--')
                plt_path = f'{dataset_name}_true_vs_predicted_ages.png'
                plt.savefig(plt_path)
                plt.close()

                wandb.log(
                    {f'{dataset_name}_scatter_plot': wandb.Image(plt_path)})

                self.test_outputs[dataset_name].clear()
                self.test_labels[dataset_name].clear()
                self.test_genders[dataset_name].clear()

    def pearson_correlation(self, pred, actual):
        return self.pearson_corr(pred, actual)

    def r_squared(self, pred, actual):
        return self.r2_score(pred, actual)
