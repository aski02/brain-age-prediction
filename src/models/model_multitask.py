import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import numpy as np
from monai.transforms import RandRotated, RandAffined, EnsureChannelFirstd
import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef, R2Score, Accuracy, AUROC
from torchmetrics import Precision, Recall, F1Score, Specificity, ConfusionMatrix
from architectures.densenet3d_gender import SupRegDenseNet as MultitaskDenseNet
from architectures.resnet3d_gender import SupRegResNet as MultitaskResNet
from architectures.densenet3d_gender import SupRegDenseNetDropout as MultitaskDenseNetDropout
from architectures.resnet3d_gender import SupRegResNetDropout as MultitaskResNetDropout
import matplotlib.pyplot as plt


class AutomaticWeightedLoss(nn.Module):
    # Taken from https://github.com/Mikoto10032/AutomaticWeightedLoss
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * \
                loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class BrainGenderModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config['dropout'] == 1:
            if self.config['model'] == "ResNet":
                self.model = MultitaskResNetDropout()
            elif self.config['model'] == "DenseNet":
                self.model = MultitaskDenseNetDropout()
        else:
            if self.config['model'] == "ResNet":
                self.model = MultitaskResNet()
            elif self.config['model'] == "DenseNet":
                self.model = MultitaskDenseNet()

        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.bce_criterion = nn.BCEWithLogitsLoss()

        if self.config.get('multitask_loss', 0) == 1:
            self.awl = AutomaticWeightedLoss(2)

        self.pearson_corr = PearsonCorrCoef(num_outputs=1)
        self.r2_score = R2Score(num_outputs=1)
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1_score = F1Score(task="binary")
        self.specificity = Specificity(task="binary")

        max_rad = np.radians(40)
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
            translate_range=(5, 5, 5),
            mode='bilinear',
            padding_mode='zeros',
            cache_grid=True
        )

        self.test_outputs = {}
        self.test_labels = {}
        self.test_genders = {}
        self.test_patient_ids = {}
        self.test_pred_genders = {}
        self.test_acc = {}
        self.test_auc = {}
        self.test_precision = {}
        self.test_recall = {}
        self.test_f1_score = {}
        self.test_specificity = {}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler} if scheduler else optimizer

    def _get_optimizer(self):
        optimizer_params = [
            {'params': self.model.parameters()}
        ]
        if self.config.get('multitask_loss', 0) == 1:
            optimizer_params.append(
                {'params': self.awl.parameters(), 'weight_decay': 0})

        if self.config['optimizer'] == 'Adam':
            return optim.Adam(optimizer_params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'AdamW':
            return optim.AdamW(optimizer_params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'SGD':
            return optim.SGD(optimizer_params, lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])

    def _get_scheduler(self, optimizer):
        if self.config['scheduler'] == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, threshold=0.05, threshold_mode='abs', verbose=True)
        elif self.config['scheduler'] == 'StepLR':
            return StepLR(optimizer, step_size=10, gamma=0.9)
        elif self.config['scheduler'] == 'CosineAnnealingLR':
            return CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        elif self.config['scheduler'] == 'CosineAnnealingWarmRestarts':
            return CosineAnnealingWarmRestarts(optimizer, T_0=35)
        return None

    def _compute_loss(self, age_loss, gender_loss):
        if self.config['multitask_loss'] == 1:
            return self.awl(age_loss, gender_loss)
        elif self.config['multitask_loss'] == 0:
            return age_loss + gender_loss

    def training_step(self, batch, batch_idx):
        images, ages, genders, scanner = batch['image'], batch['label'].float(
        ), batch['gender'].float(), batch['scanner'].int()
        age_outputs, gender_outputs = self(images)
        ages = ages.unsqueeze(1)

        age_loss = self.mae_criterion(age_outputs, ages)
        gender_loss = self.bce_criterion(gender_outputs.squeeze(), genders)

        loss_sum = self._compute_loss(age_loss, gender_loss)

        self.log('train_mae', age_loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=len(images))
        self.log('train_gender_loss', gender_loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=len(images))
        self.log('train_loss', loss_sum, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=len(images))

        return loss_sum

    def validation_step(self, batch, batch_idx):
        images, ages, genders = batch['image'], batch['label'].float(
        ), batch['gender'].float()
        age_outputs, gender_outputs = self(images)
        ages = ages.unsqueeze(1)

        age_loss = self.mae_criterion(age_outputs, ages)
        gender_loss = self.bce_criterion(gender_outputs.squeeze(), genders)
        loss_sum = self._compute_loss(age_loss, gender_loss)

        self._log_validation_metrics(
            age_outputs, ages, gender_outputs, genders, loss_sum, age_loss, gender_loss, len(images))

        return loss_sum

    def _log_validation_metrics(self, age_outputs, ages, gender_outputs, genders, loss_sum, age_loss, gender_loss, batch_size):
        r_value = self.pearson_corr(age_outputs, ages)
        r_squared = self.r2_score(age_outputs, ages)

        preds = torch.sigmoid(gender_outputs).squeeze()
        acc = self.accuracy(preds, genders.long())
        auc = self.auroc(preds, genders.long())
        precision = self.precision(preds, genders.long())
        recall = self.recall(preds, genders.long())
        f1 = self.f1_score(preds, genders.long())
        specificity = self.specificity(preds, genders.long())

        self.log('val_mae', age_loss, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log('val_gender_loss', gender_loss, on_step=False,
                 on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log('val_loss', loss_sum, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log('val_r_value', r_value, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log('val_r_squared', r_squared, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_auc', auc, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_precision', precision, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_recall', recall, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_f1_score', f1, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_specificity', specificity, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_names = ['hc', 'mogad', 'nmosd', 'ms_trio',
                         'ms_prisma', 'ms_berl', 'val', 'val_charite', 'val_openbhb']
        dataset_name = dataset_names[dataloader_idx]

        images, ages, genders, patient_ids = batch['image'], batch['label'].float(
        ), batch['gender'].float(), batch['patient_id']

        age_outputs, gender_outputs = self(images)
        ages = ages.unsqueeze(1)

        if dataset_name not in self.test_outputs:
            self.test_outputs[dataset_name] = []
            self.test_labels[dataset_name] = []
            self.test_genders[dataset_name] = []
            self.test_patient_ids[dataset_name] = []
            self.test_pred_genders[dataset_name] = []
            self.test_acc[dataset_name] = []
            self.test_auc[dataset_name] = []
            self.test_precision[dataset_name] = []
            self.test_recall[dataset_name] = []
            self.test_f1_score[dataset_name] = []
            self.test_specificity[dataset_name] = []

        self.test_outputs[dataset_name].append(age_outputs)
        self.test_labels[dataset_name].append(ages)
        self.test_genders[dataset_name].append(genders)
        self.test_patient_ids[dataset_name].append(patient_ids)

        preds = torch.sigmoid(gender_outputs).squeeze()
        acc = self.accuracy(preds, genders.long())
        auc = self.auroc(preds, genders.long())
        precision = self.precision(preds, genders.long())
        recall = self.recall(preds, genders.long())
        f1 = self.f1_score(preds, genders.long())
        specificity = self.specificity(preds, genders.long())

        self.test_pred_genders[dataset_name].append(preds)
        self.test_acc[dataset_name].append(acc)
        self.test_auc[dataset_name].append(auc)
        self.test_precision[dataset_name].append(precision)
        self.test_recall[dataset_name].append(recall)
        self.test_f1_score[dataset_name].append(f1)
        self.test_specificity[dataset_name].append(specificity)

    def on_test_epoch_end(self):
        dataset_names = ['hc', 'mogad', 'nmosd', 'ms_trio',
                         'ms_prisma', 'ms_berl', 'val', 'val_charite', 'val_openbhb']

        for dataset_name in dataset_names:
            if dataset_name in self.test_outputs:
                outputs = torch.cat(self.test_outputs[dataset_name], dim=0)
                ages = torch.cat(self.test_labels[dataset_name], dim=0)
                genders = torch.cat(self.test_genders[dataset_name], dim=0)
                predicted_genders = torch.cat(
                    self.test_pred_genders[dataset_name], dim=0)
                patient_ids = self.test_patient_ids[dataset_name]

                mae_loss = self.mae_criterion(outputs, ages)
                r_value = self.pearson_corr(outputs, ages)

                acc = torch.mean(torch.stack(self.test_acc[dataset_name]))
                auc = torch.mean(torch.stack(self.test_auc[dataset_name]))
                precision = torch.mean(torch.stack(
                    self.test_precision[dataset_name]))
                recall = torch.mean(torch.stack(
                    self.test_recall[dataset_name]))
                f1 = torch.mean(torch.stack(self.test_f1_score[dataset_name]))
                specificity = torch.mean(torch.stack(
                    self.test_specificity[dataset_name]))

                self.log(f'test_mae_{dataset_name}', mae_loss, on_step=False,
                         on_epoch=True, prog_bar=False, logger=True, batch_size=len(ages))
                self.log(f'test_r_value_{dataset_name}', r_value, on_step=False,
                         on_epoch=True, prog_bar=False, logger=True, batch_size=len(ages))
                self.log(f'test_acc_{dataset_name}', acc, on_step=False,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=len(genders))
                self.log(f'test_auc_{dataset_name}', auc, on_step=False,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=len(genders))
                self.log(f'test_precision_{dataset_name}', precision, on_step=False,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=len(genders))
                self.log(f'test_recall_{dataset_name}', recall, on_step=False,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=len(genders))
                self.log(f'test_f1_score_{dataset_name}', f1, on_step=False,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=len(genders))
                self.log(f'test_specificity_{dataset_name}', specificity, on_step=False,
                         on_epoch=True, prog_bar=True, logger=True, batch_size=len(genders))

                patient_ids_flat = [
                    pid for sublist in patient_ids for pid in sublist]

                true_ages_list = ages.squeeze().cpu().numpy().tolist()
                predicted_ages_list = outputs.squeeze().cpu().numpy().tolist()
                genders_list = genders.squeeze().cpu().numpy().tolist()
                predicted_genders_list = predicted_genders.squeeze().cpu().numpy().tolist()

                wandb.log({f'{dataset_name}_true_ages': true_ages_list})
                wandb.log(
                    {f'{dataset_name}_predicted_ages': predicted_ages_list})
                wandb.log({f'{dataset_name}_genders': genders_list})
                wandb.log(
                    {f'{dataset_name}_predicted_genders': predicted_genders_list})
                wandb.log({f'{dataset_name}_patient_ids': patient_ids_flat})

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
                self.test_pred_genders[dataset_name].clear()
                self.test_acc[dataset_name].clear()
                self.test_auc[dataset_name].clear()
                self.test_precision[dataset_name].clear()
                self.test_recall[dataset_name].clear()
                self.test_f1_score[dataset_name].clear()
                self.test_specificity[dataset_name].clear()
                self.test_patient_ids[dataset_name].clear()

    def pearson_correlation(self, pred, actual):
        return self.pearson_corr(pred, actual)

    def r_squared(self, pred, actual):
        return self.r2_score(pred, actual)
