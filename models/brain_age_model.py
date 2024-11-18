import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from monai.transforms import RandRotated, RandAffined
from torchmetrics import PearsonCorrCoef, R2Score, Accuracy, AUROC, Precision, Recall, F1Score, Specificity
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR

from models.densenet_models import SupRegDenseNet, SupRegDenseNetMultiTask
from models.resnet_models import SupRegResNet, SupRegResNetMultiTask


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * \
                loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class BrainAgeModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_name = self.config.get('model', 'ResNet')
        model_type = self.config.get('model_type', 'basic')
        dropout = self.config.get('dropout', False)
        gender_input = model_type == 'feature'

        # Select the model
        if model_name == 'ResNet':
            if model_type == 'basic' or model_type == 'feature':
                self.model = SupRegResNet(
                    name=self.config.get('backbone', 'resnet18'),
                    dropout=dropout,
                    gender_input=gender_input
                )
            elif model_type == 'multitask':
                self.model = SupRegResNetMultiTask(
                    name=self.config.get('backbone', 'resnet18'),
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        elif model_name == 'DenseNet':
            if model_type == 'basic' or model_type == 'feature':
                self.model = SupRegDenseNet(
                    dropout=dropout,
                    gender_input=gender_input
                )
            elif model_type == 'multitask':
                self.model = SupRegDenseNetMultiTask(
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Define loss functions and metrics
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.bce_criterion = nn.BCEWithLogitsLoss()

        if model_type == 'multitask' and self.config.get('multitask_loss', 0) == 1:
            self.awl = AutomaticWeightedLoss(2)

        # Metrics
        self.pearson_corr = PearsonCorrCoef()
        self.r2_score = R2Score()
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1_score = F1Score(task="binary")
        self.specificity = Specificity(task="binary")

        # Data augmentation methods
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
        self.test_patient_ids = {}
        self.test_gender_outputs = {}
        self.test_metrics = {}

        self.train_outputs = []
        self.train_targets = []
        self.train_scanners = []
        self.train_gender_outputs = []
        self.train_genders = []

        self.val_outputs = []
        self.val_targets = []
        self.val_scanners = []
        self.val_gender_outputs = []
        self.val_genders = []

        # Define which datasets to evaluate in the final test phase
        self.dataset_names = ['hc', 'mogad', 'nmosd', 'ms', 'ms_trio',
                              'ms_prisma', 'val', 'val_charite', 'val_openbhb']

    def forward(self, images, genders=None):
        if self.config['model_type'] == 'basic':
            return self.model(images)
        elif self.config['model_type'] == 'feature':
            return self.model(images, genders)
        elif self.config['model_type'] == 'multitask':
            return self.model(images)
        else:
            raise ValueError(
                f"Unknown model_type: {self.config['model_type']}")

    def configure_optimizers(self):
        optimizer_params = [{'params': self.model.parameters()}]

        if self.config.get('multitask_loss', 0) == 1 and self.config['model_type'] == 'multitask':
            optimizer_params.append(
                {'params': self.awl.parameters(), 'weight_decay': 0})

        optimizer = self._get_optimizer(optimizer_params)
        scheduler = self._get_scheduler(optimizer)
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'frequency': 1,
                        'strict': True,
                        'name': 'learning_rate'
                    }
                }
            else:
                return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer

    def _get_optimizer(self, optimizer_params):
        optimizer_name = self.config.get('optimizer', 'Adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0)

        if optimizer_name == 'Adam':
            return optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            return optim.SGD(optimizer_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_scheduler(self, optimizer):
        scheduler_name = self.config.get('scheduler', None)
        if scheduler_name == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.7,
                                     patience=10, threshold=0.05,
                                     threshold_mode='abs')
        elif scheduler_name == 'StepLR':
            return StepLR(optimizer, step_size=10, gamma=0.9)
        elif scheduler_name == 'CosineAnnealingLR':
            return CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        elif scheduler_name == 'CosineAnnealingWarmRestarts':
            return CosineAnnealingWarmRestarts(optimizer, T_0=35)
        else:
            return None

    def training_step(self, batch, batch_idx):
        images = batch['image']
        ages = batch['label'].float()
        genders = batch.get('gender', None)
        scanner = batch['scanner'].int()

        # Apply data augmentation if specified
        images = self.apply_augmentation(images)

        if self.config['model_type'] == 'basic':
            outputs, _ = self.forward(images)
            loss = self.compute_age_loss(outputs, ages, scanner)
            self.train_outputs.append(outputs.detach())
            self.train_targets.append(ages.detach())
            self.train_scanners.append(scanner.detach())
            return loss
        elif self.config['model_type'] == 'feature':
            outputs, _ = self.forward(images, genders)
            loss = self.compute_age_loss(outputs, ages, scanner)
            self.train_outputs.append(outputs.detach())
            self.train_targets.append(ages.detach())
            self.train_scanners.append(scanner.detach())
            return loss
        elif self.config['model_type'] == 'multitask':
            age_outputs, gender_outputs = self.forward(images)
            loss = self.compute_multitask_loss(
                age_outputs, ages, gender_outputs, genders)
            self.train_outputs.append(age_outputs.detach())
            self.train_targets.append(ages.detach())
            self.train_scanners.append(scanner.detach())
            self.train_gender_outputs.append(gender_outputs.detach())
            self.train_genders.append(genders.detach())
            return loss

    def on_train_epoch_end(self):
        outputs = torch.cat(self.train_outputs)
        targets = torch.cat(self.train_targets)
        scanners = torch.cat(self.train_scanners)
        train_loss = self.compute_age_loss(outputs, targets, scanners)

        mse_loss = self.mse_criterion(outputs, targets.unsqueeze(1))
        mae_loss = self.mae_criterion(outputs, targets.unsqueeze(1))
        r_value = self.pearson_corr(outputs.squeeze(), targets)
        r_squared = self.r2_score(outputs.squeeze(), targets)

        self.log('train_loss', train_loss)
        self.log('train_mse', mse_loss)
        self.log('train_mae', mae_loss)
        self.log('train_r_value', r_value)
        self.log('train_r_squared', r_squared)

        if self.config['model_type'] == 'multitask':
            gender_outputs = torch.cat(self.train_gender_outputs)
            genders = torch.cat(self.train_genders)
            preds = torch.sigmoid(gender_outputs).squeeze()
            acc = self.accuracy(preds, genders.long())
            auc = self.auroc(preds, genders.long())
            precision = self.precision(preds, genders.long())
            recall = self.recall(preds, genders.long())
            f1 = self.f1_score(preds, genders.long())
            specificity = self.specificity(preds, genders.long())

            self.log('train_acc', acc)
            self.log('train_auc', auc)
            self.log('train_precision', precision)
            self.log('train_recall', recall)
            self.log('train_f1_score', f1)
            self.log('train_specificity', specificity)

            self.train_gender_outputs = []
            self.train_genders = []

            self.accuracy.reset()
            self.auroc.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1_score.reset()
            self.specificity.reset()

        self.train_outputs = []
        self.train_targets = []
        self.train_scanners = []

        self.pearson_corr.reset()
        self.r2_score.reset()

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        ages = batch['label'].float()
        genders = batch.get('gender', None)
        scanner = batch.get('scanner', None)

        if self.config['model_type'] == 'basic':
            outputs, _ = self.forward(images)
            loss = self.compute_age_loss(outputs, ages)
            self.val_outputs.append(outputs.detach())
            self.val_targets.append(ages.detach())
            return loss
        elif self.config['model_type'] == 'feature':
            outputs, _ = self.forward(images, genders)
            loss = self.compute_age_loss(outputs, ages)
            self.val_outputs.append(outputs.detach())
            self.val_targets.append(ages.detach())
            return loss
        elif self.config['model_type'] == 'multitask':
            age_outputs, gender_outputs = self.forward(images)
            loss = self.compute_multitask_loss(
                age_outputs, ages, gender_outputs, genders)
            self.val_outputs.append(age_outputs.detach())
            self.val_targets.append(ages.detach())
            self.val_gender_outputs.append(gender_outputs.detach())
            self.val_genders.append(genders.detach())
            return loss

    def on_validation_epoch_end(self):
        # Concatenate the outputs and targets across the batches
        outputs = torch.cat(self.val_outputs)
        targets = torch.cat(self.val_targets)

        # Calculate losses based on the model type
        if self.config['model_type'] == 'multitask':
            # For multitask models, calculate and log the multitask loss
            age_outputs = outputs
            gender_outputs = torch.cat(self.val_gender_outputs)
            genders = torch.cat(self.val_genders)

            # Compute the multitask loss (age loss + gender loss)
            multitask_loss = self.compute_multitask_loss(
                age_outputs, targets, gender_outputs, genders)
            self.log('val_loss', multitask_loss)  # Log multitask loss

            # You can still calculate and log individual losses if necessary
            val_loss = self.mae_criterion(age_outputs, targets.unsqueeze(1))
            mse_loss = self.mse_criterion(age_outputs, targets.unsqueeze(1))
            self.log('val_mae', val_loss)
            self.log('val_mse', mse_loss)

            # Log additional metrics for gender classification
            preds = torch.sigmoid(gender_outputs).squeeze()
            acc = self.accuracy(preds, genders.long())
            auc = self.auroc(preds, genders.long())
            precision = self.precision(preds, genders.long())
            recall = self.recall(preds, genders.long())
            f1 = self.f1_score(preds, genders.long())
            specificity = self.specificity(preds, genders.long())

            self.log('val_acc', acc)
            self.log('val_auc', auc)
            self.log('val_precision', precision)
            self.log('val_recall', recall)
            self.log('val_f1_score', f1)
            self.log('val_specificity', specificity)

            # Reset gender-specific variables
            self.val_gender_outputs = []
            self.val_genders = []
            self.accuracy.reset()
            self.auroc.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1_score.reset()
            self.specificity.reset()

        else:
            # For 'basic' and 'feature' models, log the MAE and MSE losses
            val_loss = self.mae_criterion(outputs, targets.unsqueeze(1))
            mse_loss = self.mse_criterion(outputs, targets.unsqueeze(1))
            r_value = self.pearson_corr(outputs.squeeze(), targets)
            r_squared = self.r2_score(outputs.squeeze(), targets)

            # Log metrics for regression tasks
            self.log('val_loss', val_loss)
            self.log('val_mse', mse_loss)
            self.log('val_mae', val_loss)
            self.log('val_r_value', r_value)
            self.log('val_r_squared', r_squared)

        # Reset outputs and targets
        self.val_outputs = []
        self.val_targets = []

        # Reset correlation and R2 metrics
        self.pearson_corr.reset()
        self.r2_score.reset()

    def on_validation_epoch_end_old(self):
        outputs = torch.cat(self.val_outputs)
        targets = torch.cat(self.val_targets)
        val_loss = self.mae_criterion(outputs, targets.unsqueeze(1))

        mse_loss = self.mse_criterion(outputs, targets.unsqueeze(1))
        mae_loss = val_loss
        r_value = self.pearson_corr(outputs.squeeze(), targets)
        r_squared = self.r2_score(outputs.squeeze(), targets)

        self.log('val_loss', val_loss)
        self.log('val_mse', mse_loss)
        self.log('val_mae', mae_loss)
        self.log('val_r_value', r_value)
        self.log('val_r_squared', r_squared)

        if self.config['model_type'] == 'multitask':
            gender_outputs = torch.cat(self.val_gender_outputs)
            genders = torch.cat(self.val_genders)
            preds = torch.sigmoid(gender_outputs).squeeze()
            acc = self.accuracy(preds, genders.long())
            auc = self.auroc(preds, genders.long())
            precision = self.precision(preds, genders.long())
            recall = self.recall(preds, genders.long())
            f1 = self.f1_score(preds, genders.long())
            specificity = self.specificity(preds, genders.long())

            self.log('val_acc', acc)
            self.log('val_auc', auc)
            self.log('val_precision', precision)
            self.log('val_recall', recall)
            self.log('val_f1_score', f1)
            self.log('val_specificity', specificity)

            self.val_gender_outputs = []
            self.val_genders = []

            self.accuracy.reset()
            self.auroc.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1_score.reset()
            self.specificity.reset()

        self.val_outputs = []
        self.val_targets = []

        self.pearson_corr.reset()
        self.r2_score.reset()

    def test_step(self, batch, batch_idx, dataloader_idx):
        dataset_name = self.dataset_names[dataloader_idx]

        images = batch['image']
        ages = batch['label'].float()
        genders = batch.get('gender', None)
        patient_ids = batch.get('patient_id', None)

        if self.config['model_type'] == 'multitask':
            outputs, gender_outputs = self.forward(images)
            if dataset_name not in self.test_gender_outputs:
                self.test_gender_outputs[dataset_name] = []
                self.test_genders[dataset_name] = []
            self.test_gender_outputs[dataset_name].append(
                gender_outputs.detach())
            self.test_genders[dataset_name].append(genders.detach())
        else:
            outputs, _ = self.forward(images)

        if dataset_name not in self.test_outputs:
            self.test_outputs[dataset_name] = []
            self.test_labels[dataset_name] = []
            self.test_patient_ids[dataset_name] = []

        self.test_outputs[dataset_name].append(outputs.detach())
        self.test_labels[dataset_name].append(ages.detach())
        if patient_ids is not None:
            self.test_patient_ids[dataset_name].append(patient_ids)

    def test_step_old(self, batch, batch_idx, dataloader_idx):
        dataset_name = self.dataset_names[dataloader_idx]

        images = batch['image']
        ages = batch['label'].float()
        genders = batch.get('gender', None)
        patient_ids = batch.get('patient_id', None)

        if self.config['model_type'] == 'basic':
            outputs, _ = self.forward(images)
        elif self.config['model_type'] == 'feature':
            outputs, _ = self.forward(images, genders)
        elif self.config['model_type'] == 'multitask':
            outputs, gender_outputs = self.forward(images)
            if dataset_name not in self.test_gender_outputs:
                self.test_gender_outputs[dataset_name] = []
                self.test_genders[dataset_name] = []
            self.test_gender_outputs[dataset_name].append(
                gender_outputs.detach())
            self.test_genders[dataset_name].append(genders.detach())

        if dataset_name not in self.test_outputs:
            self.test_outputs[dataset_name] = []
            self.test_labels[dataset_name] = []
            self.test_patient_ids[dataset_name] = []
            if genders is not None:
                self.test_genders[dataset_name] = []

        self.test_outputs[dataset_name].append(outputs.detach())
        self.test_labels[dataset_name].append(ages.detach())
        if patient_ids is not None:
            self.test_patient_ids[dataset_name].append(patient_ids)

    def on_test_epoch_end(self):
        for idx, dataset_name in enumerate(self.dataset_names):
            if dataset_name not in self.test_outputs:
                continue

            outputs = torch.cat(self.test_outputs[dataset_name], dim=0)
            ages = torch.cat(self.test_labels[dataset_name], dim=0)

            print(len(outputs), len(ages))

            mae_loss = self.mae_criterion(outputs, ages.unsqueeze(1))
            r_value = self.pearson_corr(outputs.squeeze(), ages)
            self.log(f'test_mae_{dataset_name}', mae_loss)
            self.log(f'test_r_value_{dataset_name}', r_value)

            true_ages_list = ages.cpu().numpy().tolist()
            predicted_ages_list = outputs.squeeze().cpu().numpy().tolist()

            print(len(true_ages_list), len(predicted_ages_list))

            wandb.log({f'{dataset_name}_true_ages': true_ages_list})
            wandb.log({f'{dataset_name}_predicted_ages': predicted_ages_list})

            if dataset_name in self.test_genders and self.test_genders[dataset_name]:
                genders = torch.cat(self.test_genders[dataset_name], dim=0)
                genders_list = genders.cpu().numpy().tolist()
                wandb.log({f'{dataset_name}_genders': genders_list})

                print(len(genders_list))

                if self.config['model_type'] == 'multitask':
                    gender_outputs = torch.cat(
                        self.test_gender_outputs[dataset_name], dim=0)
                    preds = torch.sigmoid(gender_outputs).squeeze()
                    predicted_genders_list = preds.cpu().numpy().tolist()
                    wandb.log(
                        {f'{dataset_name}_predicted_genders': predicted_genders_list})

                    print(len(preds), len(genders.long()))

                    acc = self.accuracy(preds, genders.long())
                    auc = self.auroc(preds, genders.long())
                    precision = self.precision(preds, genders.long())
                    recall = self.recall(preds, genders.long())
                    f1 = self.f1_score(preds, genders.long())
                    specificity = self.specificity(preds, genders.long())

                    self.log(f'test_acc_{dataset_name}', acc)
                    self.log(f'test_auc_{dataset_name}', auc)
                    self.log(f'test_precision_{dataset_name}', precision)
                    self.log(f'test_recall_{dataset_name}', recall)
                    self.log(f'test_f1_score_{dataset_name}', f1)
                    self.log(f'test_specificity_{dataset_name}', specificity)

                    self.accuracy.reset()
                    self.auroc.reset()
                    self.precision.reset()
                    self.recall.reset()
                    self.f1_score.reset()
                    self.specificity.reset()

            if dataset_name in self.test_patient_ids and self.test_patient_ids[dataset_name]:
                # Flatten the list of lists
                patient_ids_list = [
                    pid for sublist in self.test_patient_ids[dataset_name] for pid in sublist]
                wandb.log({f'{dataset_name}_patient_ids': patient_ids_list})

            # Create scatter plots automatically
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

            wandb.log({f'{dataset_name}_scatter_plot': wandb.Image(plt_path)})
            os.remove(plt_path)

            self.test_outputs[dataset_name].clear()
            self.test_labels[dataset_name].clear()
            if dataset_name in self.test_genders:
                self.test_genders[dataset_name].clear()
            if dataset_name in self.test_patient_ids:
                self.test_patient_ids[dataset_name].clear()
            if self.config['model_type'] == 'multitask' and dataset_name in self.test_gender_outputs:
                self.test_gender_outputs[dataset_name].clear()

        self.pearson_corr.reset()
        self.r2_score.reset()
        if self.config['model_type'] == 'multitask':
            self.accuracy.reset()
            self.auroc.reset()
            self.precision.reset()
            self.recall.reset()
            self.f1_score.reset()
            self.specificity.reset()

    def apply_augmentation(self, images):
        augmentation = self.config.get('augmentation', 0)
        transformed = {'image': images}
        if augmentation == 1:
            transformed = self.random_rotation(transformed)
            transformed = self.random_affine(transformed)
        elif augmentation == 2:
            transformed = self.random_affine(transformed)
        elif augmentation == 3:
            transformed = self.random_rotation(transformed)
        return transformed['image']

    def apply_augmentation_old(self, images):
        augmentation = self.config.get('augmentation', 0)
        if augmentation == 1:
            # Apply both rotation and affine transformations
            transformed_images = []
            for img in images:
                transformed = {'image': img}
                transformed = self.random_rotation(transformed)
                transformed = self.random_affine(transformed)
                transformed_images.append(transformed['image'])
            images = torch.stack(transformed_images, dim=0)
        elif augmentation == 2:
            # Apply only affine transformation
            transformed_images = []
            for img in images:
                transformed = {'image': img}
                transformed = self.random_affine(transformed)
                transformed_images.append(transformed['image'])
            images = torch.stack(transformed_images, dim=0)
        elif augmentation == 3:
            # Apply only rotation
            transformed_images = []
            for img in images:
                transformed = {'image': img}
                transformed = self.random_rotation(transformed)
                transformed_images.append(transformed['image'])
            images = torch.stack(transformed_images, dim=0)
        return images

    def compute_age_loss(self, outputs, ages, scanner=None):
        ages = ages.unsqueeze(1)
        mae_loss = self.mae_criterion(outputs, ages)
        if self.config.get('loss_weighted', 0) and scanner is not None:
            weight = self.get_loss_weights(scanner)
            mae_loss = mae_loss.squeeze() * weight
            mae_loss = mae_loss.mean()
        else:
            mae_loss = mae_loss.mean()
        return mae_loss

    def compute_multitask_loss(self, age_outputs, ages, gender_outputs, genders):
        age_loss = self.mae_criterion(age_outputs, ages.unsqueeze(1)).mean()
        gender_loss = self.bce_criterion(
            gender_outputs.squeeze(), genders.float()).mean()
        if self.config.get('multitask_loss', 0) == 1:
            loss = self.awl(age_loss, gender_loss)
        else:
            loss = age_loss + gender_loss
        return loss

    def get_loss_weights(self, scanner):
        weight = torch.ones_like(scanner, dtype=torch.float32)
        loss_weighted = self.config.get('loss_weighted', 0)
        if loss_weighted == 1:
            weight[scanner == 2] = 3
            weight[scanner == 1] = 3
        elif loss_weighted == 2:
            weight[scanner == 2] = 10
            weight[scanner == 1] = 10
        elif loss_weighted == 3:
            weight[scanner == 2] = 50
            weight[scanner == 1] = 50
        return weight

    def get_test_predictions(self):
        predictions = {}
        for dataset_name in self.dataset_names:
            if dataset_name in self.test_outputs and self.test_outputs[dataset_name]:
                outputs = torch.cat(self.test_outputs[dataset_name], dim=0)
                ages = torch.cat(self.test_labels[dataset_name], dim=0)
                predictions[dataset_name] = {
                    'predicted_ages': outputs.cpu().numpy(),
                    'true_ages': ages.cpu().numpy()
                }
                if dataset_name in self.test_genders and self.test_genders[dataset_name]:
                    genders = torch.cat(self.test_genders[dataset_name], dim=0)
                    predictions[dataset_name]['genders'] = genders.cpu().numpy()
                if dataset_name in self.test_patient_ids and self.test_patient_ids[dataset_name]:
                    patient_ids = [
                        pid for sublist in self.test_patient_ids[dataset_name] for pid in sublist]
                    predictions[dataset_name]['patient_ids'] = patient_ids
        return predictions
