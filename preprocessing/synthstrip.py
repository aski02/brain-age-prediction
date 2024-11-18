import os
import subprocess
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run SynthStrip for brain mask creation.")
    parser.add_argument('--root_path', type=str, required=True, help='Root directory path for the images')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file with relative paths of input images')
    return parser.parse_args()

def main():
    args = parse_args()
    
    root_path = args.root_path
    csv_file = args.csv_file

    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        relative_path = row['Paths']
        file_path = os.path.join(root_path, relative_path)

        print(f"Processing {file_path}")

        input_dir = os.path.dirname(file_path)
        preds_dir = os.path.join(input_dir, 'brain_masks_synthstrip')

        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)

        input_file = file_path

        output_mask_name = os.path.basename(file_path).replace('.nii.gz', '_mask.nii.gz')
        output_mask_path = os.path.join(preds_dir, output_mask_name)

        synthstrip_command = f"mri_synthstrip -i {input_file} -m {output_mask_path}"

        synthstrip_process = subprocess.Popen(synthstrip_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        synthstrip_stdout, synthstrip_stderr = synthstrip_process.communicate()

        print(f"Running command: {synthstrip_command}")
        print("SynthStrip Output:", synthstrip_stdout)
        print("SynthStrip Error:", synthstrip_stderr)

    print("Processing completed.")

if __name__ == "__main__":
    main()

