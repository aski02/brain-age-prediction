import argparse
import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch import sigmoid
from monai.data import Dataset, DataLoader
from monai.transforms import EnsureTyped
from data.brain_age_data_module import BrainAgeDataModule, SqueezeDimd, CropRegiond, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, DeleteItemsd, EnsureTyped, Compose
from models.brain_age_model import BrainAgeModel
from pytorch_lightning import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description='Region-Based Brain Age Inference Script')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--atlas_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='', help='Root directory to prepend to input paths')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
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

    if args.data_root:
        input_df['Paths'] = input_df['Paths'].apply(lambda x: os.path.join(args.data_root, x))

    data_list = [{'image': path} for path in input_df['Paths']]

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
    region_ids = [rid for sublist in GPU_REGION_MAPPING.values()
                  for rid in sublist]

    predictions_dict_age = {rid: [] for rid in region_ids}
    predictions_dict_sex = {rid: [] for rid in region_ids}

    for region_id in tqdm(region_ids):
        config['region_id'] = region_id
        run_name = f"{config['run_name']}_region_{region_id}"
        model_dir_region = os.path.join(args.model_dir, run_name)
        model_path = os.path.join(model_dir_region, 'best-checkpoint.ckpt')
        if not os.path.exists(model_path):
            print(
                f'Model checkpoint not found for region {region_id} at {model_path}')
            continue

        model = BrainAgeModel.load_from_checkpoint(model_path, config=config)
        model.eval()
        model.to(torch.device(args.device))

        transforms_list = [
            LoadImaged(keys=['image'], image_only=True),
            SqueezeDimd(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            CropRegiond(keys=['image'], region_id=region_id,
                        atlas_path=args.atlas_path),
            ScaleIntensityd(keys=['image']),
            DeleteItemsd(keys=['image_meta_dict']),
            EnsureTyped(keys=['image'], track_meta=False,
                        allow_missing_keys=True)
        ]
        region_transforms = Compose(transforms_list)

        region_dataset = Dataset(data=data_list, transform=region_transforms)
        region_dataloader = DataLoader(
            region_dataset, batch_size=1, shuffle=False)

        region_predictions_age = []
        region_predictions_sex = []
        with torch.no_grad():
            for batch in region_dataloader:
                images = batch['image'].to(torch.device(args.device))
                age_output, sex_logits = model(images)
                
                sex_prob = sigmoid(sex_logits).cpu().numpy().flatten()[0]
                
                region_predictions_age.append(age_output.cpu().numpy().flatten()[0])
                region_predictions_sex.append(sex_prob)

        predictions_dict_age[region_id] = region_predictions_age
        predictions_dict_sex[region_id] = region_predictions_sex

    predicted_columns_age = pd.DataFrame(predictions_dict_age)
    predicted_columns_sex = pd.DataFrame(predictions_dict_sex)

    if len(predicted_columns_age) == len(input_df) and len(predicted_columns_sex) == len(input_df):
        input_df = pd.concat(
            [input_df, predicted_columns_age.add_prefix('Predicted_Age_Region_'),
             predicted_columns_sex.add_prefix('Predicted_Sex_Region_')], axis=1)

    input_df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()

