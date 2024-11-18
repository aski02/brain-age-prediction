import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    DeleteItemsd,
    EnsureTyped,
    RandRotated,
    RandAffined,
)
from monai.data import CacheDataset, Dataset
import pytorch_lightning as pl
from monai.transforms import MapTransform
import nibabel as nib
from nilearn.image import resample_to_img
from .dataset_loader import dataset_loader
from torch.utils.data._utils.collate import default_collate
import os


def custom_collate(batch):
    patient_ids = [item.pop('patient_id') for item in batch]
    collated_batch = default_collate(batch)
    collated_batch['patient_id'] = patient_ids
    return collated_batch


class SqueezeDimd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].squeeze()
        return d


class CropRegiond(MapTransform):
    def __init__(self, keys, region_id, atlas_path):
        super().__init__(keys)
        self.region_id = region_id
        self.affine_matrix = torch.tensor([
            [-1.,  0.,  0.,  90.],
            [0.,  1.,  0., -126.],
            [0.,  0.,  1.,  -72.],
            [0.,  0.,  0.,   1.]
        ], dtype=torch.float32)

        atlas_img = nib.load(atlas_path)

        dummy_image = nib.Nifti1Image(torch.zeros(
            (182, 218, 182), dtype=torch.float32).numpy(), self.affine_matrix.numpy())
        self.resampled_atlas = resample_to_img(
            source_img=atlas_img, target_img=dummy_image, interpolation='nearest')

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float32)

            if image.ndim == 3:
                image = image.unsqueeze(0)

            # Get the atlas data and create a mask
            resampled_atlas_data = torch.tensor(
                self.resampled_atlas.get_fdata(), dtype=torch.float32)
            mask = (resampled_atlas_data == self.region_id).float()

            # Apply the mask to the image
            masked_image_data = image * mask

            # Find the bounding box coordinates of the region
            coords = torch.nonzero(mask)
            if coords.size(0) == 0:
                raise ValueError("No region found in the mask")

            x_min, y_min, z_min = coords.min(dim=0)[0]
            x_max, y_max, z_max = coords.max(dim=0)[0] + 1

            # Crop the image based on the bounding box
            sub_image_data = masked_image_data[:,
                                               x_min:x_max, y_min:y_max, z_min:z_max]

            # Padding if any dimension is smaller than 32
            pad_sizes = []
            for dim in range(3):
                size = sub_image_data.shape[dim + 1]
                if size < 32:
                    pad_before = (32 - size) // 2
                    pad_after = 32 - size - pad_before
                    pad_sizes.extend([pad_before, pad_after])
                else:
                    pad_sizes.extend([0, 0])

            # Apply padding
            if any(pad_sizes):
                sub_image_data = torch.nn.functional.pad(
                    sub_image_data, pad_sizes[::-1], mode='constant', value=0)

            d[key] = sub_image_data

        return d


class CropImaged(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.global_max_x, self.global_min_x = 166, 12
        self.global_max_y, self.global_min_y = 206, 12
        self.global_max_z, self.global_min_z = 161, 0

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key][:,
                            self.global_min_x:self.global_max_x + 1,
                            self.global_min_y:self.global_max_y + 1,
                            self.global_min_z:self.global_max_z + 1
                            ]
        return d


class BrainAgeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def create_transforms(self):
        train_transforms_list = [
            LoadImaged(keys=["image"], image_only=True),
            SqueezeDimd(keys=["image"]),
            EnsureChannelFirstd(keys=["image"])
        ]

        val_transforms_list = train_transforms_list.copy()

        crop_type = self.config.get('crop_type')
        if crop_type == 'region':
            region_id = self.config['region_id']
            atlas_type = self.config.get('atlas_type', 'neuromorphometrics')
            atlas_threshold = self.config.get('atlas_threshold', 0)
            if atlas_type == 'neuromorphometrics':
                atlas_path = self.config['atlas_path']
            elif atlas_type == 'mni':
                atlas_base_path = self.config['atlas_base_path']
                atlas_paths = {
                    0: os.path.join(atlas_base_path, 'MNI-maxprob-thr0-1mm.nii.gz'),
                    25: os.path.join(atlas_base_path, 'MNI-maxprob-thr25-1mm.nii.gz'),
                    50: os.path.join(atlas_base_path, 'MNI-maxprob-thr50-1mm.nii.gz'),
                }
                atlas_path = atlas_paths.get(atlas_threshold)
                if atlas_path is None:
                    raise ValueError(
                        f"Unsupported atlas_threshold: {atlas_threshold}")
            else:
                raise ValueError(f"Unsupported atlas_type: {atlas_type}")
            crop_transform = CropRegiond(
                keys=["image"],
                region_id=region_id,
                atlas_path=atlas_path
            )
            train_transforms_list.append(crop_transform)
            val_transforms_list.append(crop_transform)
        elif crop_type == 'image':
            crop_transform = CropImaged(keys=["image"])
            train_transforms_list.append(crop_transform)
            val_transforms_list.append(crop_transform)

        train_transforms_list.extend([
            ScaleIntensityd(keys=["image"]),
            DeleteItemsd(keys=["image_meta_dict"]),
            EnsureTyped(keys=["image", "label"], track_meta=False)
        ])

        val_transforms_list.extend([
            ScaleIntensityd(keys=["image"]),
            DeleteItemsd(keys=["image_meta_dict"]),
            EnsureTyped(keys=["image", "label"], track_meta=False)
        ])

        train_transforms = Compose(train_transforms_list)
        val_transforms = Compose(val_transforms_list)

        return train_transforms, val_transforms

    def prepare_data(self):
        # Load OpenBHB dataset
        self.openbhb_dataset = dataset_loader(
            dataset_name="OpenBHB",
            train_csv_path=self.config['train_csv_path'],
            val_csv_path=self.config['val_csv_path'],
            test_csv_path=self.config['test_csv_path'],
            data_root_path=self.config['data_root_path']
        )
        openbhb_train_data, openbhb_val_data = self.openbhb_dataset.get_data()

        # Load Charite dataset
        self.charite_dataset = dataset_loader(
            dataset_name="Charite",
            vims_data_path=self.config['vims_csv_path'],
            berl_data_path=self.config['berl_csv_path'],
            additional_data_path=self.config['additional_csv_path'],
            ms_data_path=self.config['ms_csv_path'],
            berl_ms_data_path=self.config['berl_ms_csv_path'],
            data_root_path=self.config['data_root_path']
        )
        charite_data = self.charite_dataset.get_data()
        (prisma_train, prisma_val, trio, hc, berl_mogad,
         berl_nmosd, ms, add_trio, add_prisma, ms_trio,
         ms_prisma) = charite_data

        add_prisma_split = add_prisma.sample(n=8, random_state=24)
        add_prisma_remaining = add_prisma.drop(add_prisma_split.index)
        val_charite = pd.concat(
            [prisma_val, add_prisma_split], ignore_index=True)

        data_module_type = self.config.get('data_module_type', 'default')

        if data_module_type in ['default']:
            if self.config.get('loss_weighted', 0):
                self.val_data = val_charite
            else:
                self.val_data = pd.concat(
                    [openbhb_val_data, val_charite], ignore_index=True)
            self.train_data = pd.concat([
                openbhb_train_data, prisma_train, trio, add_trio, add_prisma_remaining
            ], ignore_index=True)
        elif data_module_type == 'openbhb':
            self.val_data = openbhb_val_data
            self.train_data = openbhb_train_data
            self.test_val = pd.concat(
                [openbhb_val_data, prisma_val, add_prisma_split], ignore_index=True)
            self.test_val_charite = pd.concat(
                [prisma_val, add_prisma_split], ignore_index=True)
            self.test_val_openbhb = openbhb_val_data

        # Test datasets
        self.test_data_hc = hc
        self.test_data_mogad = berl_mogad
        self.test_data_nmosd = berl_nmosd
        self.test_data_ms_trio = ms_trio
        self.test_data_ms_prisma = ms_prisma
        self.test_data_ms = ms
        self.test_val = self.val_data
        self.test_val_charite = val_charite
        self.test_val_openbhb = openbhb_val_data

    def setup(self, stage=None):
        train_transforms, val_transforms = self.create_transforms()

        def prepare_files(data):
            return [{
                "image": path,
                "label": float(label),
                "gender": float(gender),
                "scanner": scanner,
                "patient_id": patient_id
            } for path, label, gender, scanner, patient_id in zip(
                data['Loc'], data['Age'], data['Gender'], data['Scanner'], data['patient_id']
            )]

        self.train_dataset = CacheDataset(
            data=prepare_files(self.train_data),
            transform=train_transforms,
            cache_rate=self.config.get('cache_rate', 1.0),
            runtime_cache="processes",
            num_workers=0,
            copy_cache=False
        )
        self.val_dataset = CacheDataset(
            data=prepare_files(self.val_data),
            transform=val_transforms,
            cache_rate=self.config.get('cache_rate', 1.0),
            runtime_cache="processes",
            num_workers=0,
            copy_cache=False
        )
        self.test_dataset_hc = Dataset(data=prepare_files(
            self.test_data_hc), transform=val_transforms)
        self.test_dataset_mogad = Dataset(data=prepare_files(
            self.test_data_mogad), transform=val_transforms)
        self.test_dataset_nmosd = Dataset(data=prepare_files(
            self.test_data_nmosd), transform=val_transforms)
        self.test_dataset_ms_trio = Dataset(data=prepare_files(
            self.test_data_ms_trio), transform=val_transforms)
        self.test_dataset_ms_prisma = Dataset(data=prepare_files(
            self.test_data_ms_prisma), transform=val_transforms)
        self.test_dataset_ms = Dataset(data=prepare_files(
            self.test_data_ms), transform=val_transforms)
        self.test_dataset_val = Dataset(data=prepare_files(
            self.test_val), transform=val_transforms)
        self.test_dataset_val_charite = Dataset(data=prepare_files(
            self.test_val_charite), transform=val_transforms)
        self.test_dataset_val_openbhb = Dataset(data=prepare_files(
            self.test_val_openbhb), transform=val_transforms)

        self.train_sample_weights = self.compute_sample_weights(
            self.train_data['Age'])

    def compute_sample_weights(self, ages):
        age_bins = self.config.get(
            'age_bins', list(range(5, 80, 3)) + [80, 90])
        bin_labels = range(len(age_bins) - 1)
        age_bin_indices = pd.cut(
            ages, bins=age_bins, labels=bin_labels, include_lowest=True)
        bin_counts = age_bin_indices.value_counts().sort_index()
        total_count = len(ages)
        weights = total_count / bin_counts.reindex(age_bin_indices).values
        weights /= weights.sum()
        return weights

    def train_dataloader(self):
        if self.config.get('even_distribution', 1):
            sampler = WeightedRandomSampler(
                weights=self.train_sample_weights,
                num_samples=len(self.train_sample_weights),
                replacement=True
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=custom_collate
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        batch_size = self.config['batch_size']
        return [
            DataLoader(self.test_dataset_hc,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_mogad,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_nmosd,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_ms,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_ms_trio,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_ms_prisma,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_val,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_val_charite,
                       batch_size=batch_size, shuffle=False),
            DataLoader(self.test_dataset_val_openbhb,
                       batch_size=batch_size, shuffle=False),
        ]
        # return {
        #     'hc': DataLoader(self.test_dataset_hc, batch_size=batch_size, shuffle=False),
        #     'mogad': DataLoader(self.test_dataset_mogad, batch_size=batch_size, shuffle=False),
        #     'nmosd': DataLoader(self.test_dataset_nmosd, batch_size=batch_size, shuffle=False),
        #     'ms': DataLoader(self.test_dataset_ms, batch_size=batch_size, shuffle=False),
        #     'ms_trio': DataLoader(self.test_dataset_ms_trio, batch_size=batch_size, shuffle=False),
        #     'ms_prisma': DataLoader(self.test_dataset_ms_prisma, batch_size=batch_size, shuffle=False),
        #     'val': DataLoader(self.test_dataset_val, batch_size=batch_size, shuffle=False),
        #     'val_charite': DataLoader(self.test_dataset_val_charite, batch_size=batch_size, shuffle=False),
        #     'val_openbhb': DataLoader(self.test_dataset_val_openbhb, batch_size=batch_size, shuffle=False),
        # }
