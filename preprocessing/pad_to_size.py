import logging
import os
from tqdm import tqdm
import SimpleITK as sitk
import argparse
import sys
from random import randint
import pandas as pd

def setup_logger():
    logger = logging.getLogger(__name__)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False
    log_file = os.path.join('logs/logs_pad.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) 
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Pad 2dx3 images to a given size, with random image location")
    parser.add_argument('--mri_folder', type=str, required=True, help='Path to the MRI folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--target_size', type=int, required=True, help='Target size to pad to')

    args = parser.parse_args()

    mri_folder = args.mri_folder
    output_dir = args.output_dir
    target_size = args.target_size

    logger.info(f"Starting parameter logging")

    logger.info(f"MRI_FOLDER: {mri_folder}")
    logger.info(f"OUTPUT_DIR: {output_dir}")
    logger.info(f"TARGET_SIZE: {target_size}")

    os.makedirs(output_dir, exist_ok=True)

    mri_files = [os.path.join(mri_folder, f) for f in os.listdir(mri_folder) 
                if f.endswith('.nii.gz')]

    logger.info(f"Found {len(mri_files)} mri files")

    failed_images = []

    for mri_file in tqdm(mri_files, desc="Proccessing file", unit="file"):
        logger.info(f"........Starting process for {mri_file}")
        try:
            image = sitk.ReadImage(mri_file)
            size = image.GetSize()
            if len(size) != 3 or size[2] != 3:
                raise ValueError("Image does not have exactly 3 slices")
            orig_w, orig_h, _ = size
            if orig_w != orig_h:
                raise ValueError("Image is not square")
            if orig_w > target_size or orig_h > target_size:
                raise ValueError("Original image is larger than target size")
            
            padded = sitk.Image([target_size, target_size, 3], image.GetPixelIDValue())

            max_offset_x = target_size - orig_w
            max_offset_y = target_size - orig_h
            offset_x = randint(0, max_offset_x)
            offset_y = randint(0, max_offset_y)

            padded = sitk.Paste(padded, image, image.GetSize(), destinationIndex=[offset_x, offset_y, 0], sourceIndex=[0, 0, 0])

            padded.SetSpacing(image.GetSpacing())
            padded.SetDirection(image.GetDirection())
            new_origin = list(image.GetOrigin())
            new_origin[0] -= offset_x * image.GetSpacing()[0]
            new_origin[1] -= offset_y * image.GetSpacing()[1]
            padded.SetOrigin(new_origin)

            output_file = os.path.join(output_dir, os.path.basename(mri_file))
            sitk.WriteImage(padded, output_file)
            logger.info(f"Processed {mri_file} to {output_file}")

        except Exception as e:
            logger.error(f"Error processing {mri_file}: {str(e)}")
            failed_images.append((mri_file, str(e)))
            continue

    csv_path = os.path.join(output_dir, 'failed_images.csv')
    if failed_images:
        df = pd.DataFrame(failed_images, columns=['file_path', 'error'])
        df.to_csv(csv_path, index=False)
    else:
        # Create an empty CSV with headers if no failures
        pd.DataFrame(columns=['file_path', 'error']).to_csv(csv_path, index=False)
    logger.info(f"Failed images recorded in {csv_path}")

    logger.info(f"Successfully processed {len(mri_files) - len(failed_images)} out of {len(mri_files)} images")
    if failed_images:
        logger.warning(f"{len(failed_images)} images failed processing")