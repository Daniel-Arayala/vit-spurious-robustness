import argparse
import logging
import os
import shutil

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="[%(asctime)s] - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def parse_args():
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', '-idir', type=str, required=True,
                        help='Directory with all the eye fundus images. '
                             'Must contain only imagens in the same format.')
    parser.add_argument('--img-annot-file', '-annot', type=str, required=True,
                        help='Path to the file containing the diabetic retinopathy '
                             'severity annotations for each image.')
    parser.add_argument('--img-ext', '-ie', type=str, required=False, default='jpeg',
                        help='Format of the input image.')
    logger.info('Parsing command-line arguments')
    return parser.parse_args()


def split_severities_per_folder(df_img_info, img_folder):
    for level, group in df_img_info.groupby('level'):
        severity_level_path = os.path.join(img_folder, str(level))
        os.makedirs(severity_level_path, exist_ok=True)
        for src_img_path in tqdm(group['image'].values, desc=f'Processing Severity {level}'):
            if os.path.exists(src_img_path):
                try:
                    shutil.move(src_img_path, severity_level_path)
                except FileNotFoundError:
                    logger.warning(f'Image {src_img_path} was not found and will be skipped')


def main():
    args = parse_args()
    # Reading the labels data
    logger.info('Detecting image format')
    df_img_info = pd.read_csv(args.img_annot_file)

    df_img_info = df_img_info[['image', 'level']]
    df_img_info['image'] = df_img_info['image'].transform(
        lambda file_name: os.path.join(args.img_dir, file_name + '.' + args.img_ext))
    split_severities_per_folder(df_img_info, args.img_dir)
    logger.info(f'The folder structure was created under {args.img_dir}')


if __name__ == '__main__':
    main()
