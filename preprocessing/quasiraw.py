import os
import pandas as pd
import argparse
from brainprep.workflow.quasiraw import brainprep_quasiraw

def parse_args():
    parser = argparse.ArgumentParser(description="Run QuasiRaw preprocessing.")
    parser.add_argument('--root_path', type=str, required=True, help='Root directory path for the images')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file with paths to scans and masks')
    return parser.parse_args()

def process_files(csv_file_path, base_dir):
    df = pd.read_csv(csv_file_path)

    for index, row in df.iterrows():
        try:
            scan_path = os.path.join(base_dir, row['Scan'])
            mask_path = os.path.join(base_dir, row['Mask'])

            if not os.path.exists(scan_path):
                raise FileNotFoundError(f"Scan file not found: {scan_path}")

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            scan_dir = os.path.dirname(scan_path)
            quasiraw_outdir = os.path.join(scan_dir, 'quasiraw')

            if not os.path.exists(quasiraw_outdir):
                os.makedirs(quasiraw_outdir)

            brainprep_quasiraw(
                anatomical=scan_path,
                mask=mask_path,
                outdir=quasiraw_outdir,
                target=None,
                no_bids=True
            )

            print(f"Pre-processing complete for {scan_path}")

        except Exception as e:
            print(f"An error occurred while processing {row['Scan']}: {e}")

    print("All files have been processed.")

def main():
    args = parse_args()
    process_files(csv_file_path=args.csv_file, base_dir=args.root_path)

if __name__ == "__main__":
    main()

