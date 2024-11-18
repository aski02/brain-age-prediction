import argparse
import os
import yaml
import torch
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import EnsureTyped
from data.brain_age_data_module import BrainAgeDataModule
from models.brain_age_model import BrainAgeModel
from pytorch_lightning import seed_everything
from torch import sigmoid


def parse_args():
    parser = argparse.ArgumentParser(description='Whole Brain Age Inference Script')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_root', type=str, default='', help='Root path to prepend to image paths in the input CSV')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    seed = config.get('seed', 42)
    seed_everything(seed, workers=True)

    input_df = pd.read_csv(args.input_csv)
    if 'Paths' not in input_df.columns:
        print('Error: "Paths" column not found in input CSV file.')
        return

    data_root = args.data_root
    if data_root:
        input_df['Paths'] = input_df['Paths'].apply(lambda x: os.path.join(data_root, x))

    data_list = [{'image': path} for path in input_df['Paths']]

    data_module = BrainAgeDataModule(config)
    train_transforms, val_transforms = data_module.create_transforms()
    transforms = val_transforms

    for transform in transforms.transforms:
        if isinstance(transform, EnsureTyped):
            transform.allow_missing_keys = True

    dataset = Dataset(data=data_list, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device(args.device)
    model_path = os.path.join(args.model_dir, 'best-checkpoint.ckpt')
    model = BrainAgeModel.load_from_checkpoint(model_path, config=config)
    model.eval()
    model.to(device)

    predictions_age = []
    predictions_sex = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            age_output, sex_logits = model(images)
            
            predictions_age.append(age_output.cpu().numpy().flatten()[0])
            
            sex_prob = sigmoid(sex_logits).cpu().numpy().flatten()[0]
            predictions_sex.append(sex_prob)

    input_df['Predicted_Age'] = predictions_age
    input_df['Predicted_Sex'] = predictions_sex

    input_df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()

