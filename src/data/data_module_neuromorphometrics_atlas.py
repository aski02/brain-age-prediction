import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, DeleteItemsd, EnsureTyped, AddChanneld, RandRotated, RandAffined
from monai.data import CacheDataset, Dataset
import pytorch_lightning as pl
from monai.transforms import MapTransform
import nibabel as nib
from nilearn.image import resample_img, resample_to_img
from .dataset_loader import dataset_loader
from torch.utils.data._utils.collate import default_collate


def custom_collate(batch):
    patient_ids = [item['patient_id'] for item in batch]
    for item in batch:
        del item['patient_id']

    collated_batch = default_collate(batch)
    collated_batch['patient_id'] = patient_ids

    return collated_batch


class SqueezeDimd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if d[key].ndim > 3:
                d[key] = d[key].squeeze()
        return d


class CropRegiond(MapTransform):
    def __init__(self, keys, region_id):
        super().__init__(keys)
        self.region_id = region_id
        self.affine_matrix = np.array([
            [-1.,  0.,  0.,  90.],
            [0.,  1.,  0., -126.],
            [0.,  0.,  1.,  -72.],
            [0.,  0.,  0.,   1.]
        ])
        atlas_img = nib.load('../labels_Neuromorphometrics.nii')
        dummy_image = nib.Nifti1Image(
            np.zeros((182, 218, 182)), self.affine_matrix)
        self.resampled_atlas = resample_to_img(
            source_img=atlas_img, target_img=dummy_image, interpolation='nearest')

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

            resampled_atlas_data = self.resampled_atlas.get_fdata()
            mask = (resampled_atlas_data == self.region_id).astype(int)

            masked_image_data = image * mask
            coords = np.argwhere(mask)
            if coords.size == 0:
                raise ValueError("No region found in the mask")
            x_min, y_min, z_min = coords.min(axis=0)
            x_max, y_max, z_max = coords.max(axis=0) + 1
            sub_image_data = masked_image_data[:,
                                               x_min:x_max, y_min:y_max, z_min:z_max]

            pad_sizes = [(0, 0)]
            for dim in range(3):
                size = sub_image_data.shape[dim + 1]
                if size < 32:
                    pad_before = (32 - size) // 2
                    pad_after = 32 - size - pad_before
                    pad_sizes.append((pad_before, pad_after))
                else:
                    pad_sizes.append((0, 0))

            sub_image_data = np.pad(
                sub_image_data, pad_sizes, mode='constant', constant_values=0)
            d[key] = sub_image_data
        return d


class BrainGenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def create_transforms(self):
        region_id = self.config['region_id']
        atlas_threshold = self.config['atlas_threshold']
        train_transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            SqueezeDimd(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            CropRegiond(keys=["image"], region_id=region_id),
            ScaleIntensityd(keys=["image"]),
            DeleteItemsd(keys=["image_meta_dict"]),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ])
        val_transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            SqueezeDimd(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            CropRegiond(keys=["image"], region_id=region_id),
            ScaleIntensityd(keys=["image"]),
            DeleteItemsd(keys=["image_meta_dict"]),
            EnsureTyped(keys=["image", "label"], track_meta=False)
        ])
        return train_transforms, val_transforms

    def prepare_data(self):
        self.openbhb_dataset = dataset_loader(
            dataset_name="OpenBHB", train_csv_path=self.config['train_csv_path'], val_csv_path=self.config['val_csv_path'], test_csv_path=self.config['test_csv_path'])
        self.train_data, val_openbhb = self.openbhb_dataset.get_data()

        self.charite_dataset = dataset_loader(dataset_name="Charite", vims_data_path=self.config['vims_csv_path'], berl_data_path=self.config['berl_csv_path'],
                                              additional_data_path=self.config['additional_csv_path'], ms_data_path=self.config['ms_csv_path'], berl_ms_data_path=self.config['berl_ms_csv_path'])
        prisma_train, prisma_val, trio, berl_hc, berl_mogad, berl_nmosd, add_trio, add_prisma, ms_trio, ms_prisma, ms_berl = self.charite_dataset.get_data()

        add_prisma_split = add_prisma.sample(n=8, random_state=24)
        add_prisma_remaining = add_prisma.drop(add_prisma_split.index)

        val_charite = pd.concat(
            [prisma_val, add_prisma_split], ignore_index=True)

        self.val_data = pd.concat(
            [val_openbhb, val_charite], ignore_index=True)
        self.train_data = pd.concat(
            [self.train_data, prisma_train, trio, add_trio, add_prisma_remaining], ignore_index=True)
        self.test_data_hc = berl_hc
        self.test_data_mogad = berl_mogad
        self.test_data_nmosd = berl_nmosd
        self.test_data_ms_trio = ms_trio
        self.test_data_ms_prisma = ms_prisma
        self.test_data_ms_berl = ms_berl
        self.test_val = self.val_data
        self.test_val_charite = val_charite
        self.test_val_openbhb = val_openbhb

    def setup(self, stage=None):
        train_transforms, val_transforms = self.create_transforms()

        train_files = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                       for path, label, gender, scanner, patient_id in zip(self.train_data['Loc'], self.train_data['Age'], self.train_data['Gender'], self.train_data['Scanner'], self.train_data['patient_id'])]

        val_files = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                     for path, label, gender, scanner, patient_id in zip(self.val_data['Loc'], self.val_data['Age'], self.val_data['Gender'], self.val_data['Scanner'], self.val_data['patient_id'])]

        test_files_hc = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                         for path, label, gender, scanner, patient_id in zip(self.test_data_hc['Loc'], self.test_data_hc['Age'], self.test_data_hc['Gender'], self.test_data_hc['Scanner'], self.test_data_hc['patient_id'])]

        test_files_mogad = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                            for path, label, gender, scanner, patient_id in zip(self.test_data_mogad['Loc'], self.test_data_mogad['Age'], self.test_data_mogad['Gender'], self.test_data_mogad['Scanner'], self.test_data_mogad['patient_id'])]

        test_files_nmosd = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                            for path, label, gender, scanner, patient_id in zip(self.test_data_nmosd['Loc'], self.test_data_nmosd['Age'], self.test_data_nmosd['Gender'], self.test_data_nmosd['Scanner'], self.test_data_nmosd['patient_id'])]

        test_files_ms_trio = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                              for path, label, gender, scanner, patient_id in zip(self.test_data_ms_trio['Loc'], self.test_data_ms_trio['Age'], self.test_data_ms_trio['Gender'], self.test_data_ms_trio['Scanner'], self.test_data_ms_trio['patient_id'])]

        test_files_ms_prisma = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                                for path, label, gender, scanner, patient_id in zip(self.test_data_ms_prisma['Loc'], self.test_data_ms_prisma['Age'], self.test_data_ms_prisma['Gender'], self.test_data_ms_prisma['Scanner'], self.test_data_ms_prisma['patient_id'])]

        test_files_ms_berl = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                              for path, label, gender, scanner, patient_id in zip(self.test_data_ms_berl['Loc'], self.test_data_ms_berl['Age'], self.test_data_ms_berl['Gender'], self.test_data_ms_berl['Scanner'], self.test_data_ms_berl['patient_id'])]

        test_files_val = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                          for path, label, gender, scanner, patient_id in zip(self.test_val['Loc'], self.test_val['Age'], self.test_val['Gender'], self.test_val['Scanner'], self.test_val['patient_id'])]

        test_files_val_charite = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                                  for path, label, gender, scanner, patient_id in zip(self.test_val_charite['Loc'], self.test_val_charite['Age'], self.test_val_charite['Gender'], self.test_val_charite['Scanner'], self.test_val_charite['patient_id'])]

        test_files_val_openbhb = [{"image": path, "label": float(label), "gender": float(gender), "scanner": scanner, "patient_id": patient_id}
                                  for path, label, gender, scanner, patient_id in zip(self.test_val_openbhb['Loc'], self.test_val_openbhb['Age'], self.test_val_openbhb['Gender'], self.test_val_openbhb['Scanner'], self.test_val_openbhb['patient_id'])]

        self.train_dataset = CacheDataset(data=train_files, transform=train_transforms,
                                          cache_rate=1.0, runtime_cache="processes", num_workers=4, copy_cache=False)
        self.val_dataset = CacheDataset(data=val_files, transform=val_transforms,
                                        cache_rate=1.0, runtime_cache="processes", num_workers=4, copy_cache=False)
        self.test_dataset_hc = Dataset(
            data=test_files_hc, transform=val_transforms)
        self.test_dataset_mogad = Dataset(
            data=test_files_mogad, transform=val_transforms)
        self.test_dataset_nmosd = Dataset(
            data=test_files_nmosd, transform=val_transforms)
        self.test_dataset_ms_trio = Dataset(
            data=test_files_ms_trio, transform=val_transforms)
        self.test_dataset_ms_prisma = Dataset(
            data=test_files_ms_prisma, transform=val_transforms)
        self.test_dataset_ms_berl = Dataset(
            data=test_files_ms_berl, transform=val_transforms)
        self.test_dataset_val = Dataset(
            data=test_files_val, transform=val_transforms)
        self.test_dataset_val_charite = Dataset(
            data=test_files_val_charite, transform=val_transforms)
        self.test_dataset_val_openbhb = Dataset(
            data=test_files_val_openbhb, transform=val_transforms)

        self.train_sample_weights = self.compute_sample_weights(
            self.train_data['Age'])

    def compute_sample_weights(self, ages):
        age_bins = list(range(5, 80, 3)) + [80, 90]
        bin_labels = range(len(age_bins) - 1)
        age_bin_indices = pd.cut(
            ages, bins=age_bins, labels=bin_labels, include_lowest=True)
        bin_counts = age_bin_indices.value_counts().sort_index()
        total_count = len(ages)

        weights = total_count / bin_counts.reindex(age_bin_indices).values
        weights = weights / weights.sum()

        return weights

    def train_dataloader(self):
        if self.config['even_distribution'] == 1:
            sampler = WeightedRandomSampler(weights=self.train_sample_weights, num_samples=len(
                self.train_sample_weights), replacement=True)
            loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], sampler=sampler,
                                num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=custom_collate)
        else:
            loader = DataLoader(
                self.train_dataset, batch_size=self.config['batch_size'], num_workers=4, pin_memory=True, persistent_workers=True, shuffle=True)
        return loader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return {
            'hc': DataLoader(self.test_dataset_hc, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'mogad': DataLoader(self.test_dataset_mogad, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'nmosd': DataLoader(self.test_dataset_nmosd, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'ms_trio': DataLoader(self.test_dataset_ms_trio, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'ms_prisma': DataLoader(self.test_dataset_ms_prisma, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'ms_berl': DataLoader(self.test_dataset_ms_berl, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'val': DataLoader(self.test_dataset_val, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'val_charite': DataLoader(self.test_dataset_val_charite, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
            'val_openbhb': DataLoader(self.test_dataset_val_openbhb, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        }
