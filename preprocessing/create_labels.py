import logging
import os
from tqdm import tqdm
import argparse
import sys
import pandas as pd
from pathlib import Path
import shutil


def setup_logger():
    logger = logging.getLogger(__name__)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False
    log_file = os.path.join('logs/logs_create_labels.log')
    
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

def copy_files_from_folder(folder_path, output_dir, individual_value, folder_type):
    """Copy files from a specific folder and return file data"""
    logger = logging.getLogger(__name__)
    file_data = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        logger.warning(f"{folder_type} folder does not exist: {folder_path}")
        return file_data
    
    files = [f for f in folder_path.iterdir() if f.is_file()]
    logger.info(f"Found {len(files)} files in {folder_type} folder")
    
    for file_path in tqdm(files, desc=f"Copying {folder_type} files"):
        try:
            source_folder_name = file_path.parent.name

            new_filename = f"{source_folder_name}_{file_path.name}"

            dest_path = Path(output_dir) / new_filename

            if dest_path.exists():
                logger.warning(f"File already exists in output directory: {dest_path}")
                continue
            
            shutil.copy2(file_path, dest_path)
            
            file_data.append({
                'filename': file_path.name,
                'individual': individual_value
            })
            
            logger.debug(f"Copied {file_path.name} from {folder_type} folder")
            
        except Exception as e:
            logger.error(f"Error copying file {file_path.name}: {str(e)}")
    
    return file_data


def create_csv_file(file_data, output_dir):
    """Create CSV file with file labels"""
    logger = logging.getLogger(__name__)
    
    if not file_data:
        logger.warning("No files were processed")
        return
    
    df = pd.DataFrame(file_data)
    filtered_df = df[df['filename'].str.contains('.nii.gz', na=False)]
    csv_path = Path(output_dir) / 'labels.csv'
    filtered_df.to_csv(csv_path, index=False)
    
    real_count = len([f for f in file_data if f['individual'] == 1])
    aug_count = len([f for f in file_data if f['individual'] == 0])
    
    logger.info(f"Created CSV file with {len(file_data)} entries: {csv_path}")
    logger.info(f"Real files (individual=1): {real_count}")
    logger.info(f"Aug files (individual=0): {aug_count}")


if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Extracts only the merged nodes from a folder of augmented nodes")
    parser.add_argument('--real_folder', type=str, required=True, help='Path to the MRI folder with real individual nodes')
    parser.add_argument('--aug_folder', type=str, required=True, help='Path to the MRI folder with augmented nodes')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()

    real_folder = args.real_folder
    aug_folder = args.aug_folder
    output_dir = args.output_dir

    logger.info(f"Starting parameter logging")

    logger.info(f"REAL_FOLDER: {real_folder}")
    logger.info(f"AUG_FOLDER: {aug_folder}")
    logger.info(f"OUTPUT_DIR: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info("Processing real folder files...")
        real_file_data = copy_files_from_folder(real_folder, output_dir, 1, "real")
        
        logger.info("Processing aug folder files...")
        aug_file_data = copy_files_from_folder(aug_folder, output_dir, 0, "aug")
       
        all_file_data = real_file_data + aug_file_data    

        create_csv_file(all_file_data, output_dir)
        
        logger.info(f"Process completed successfully. Total files processed: {len(all_file_data)}")
        
    except Exception as e:
        logger.error(f"Process failed with error: {str(e)}")
