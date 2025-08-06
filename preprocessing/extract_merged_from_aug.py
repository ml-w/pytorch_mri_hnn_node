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
    log_file = os.path.join('logs/logs_extr_merged.log')
    
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

    parser = argparse.ArgumentParser(description="Extracts only the merged nodes from a folder of augmented nodes")
    parser.add_argument('--mri_folder', type=str, required=True, help='Path to the MRI folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()

    mri_folder = args.mri_folder
    output_dir = args.output_dir
    csv_path = args.csv_path

    logger.info(f"Starting parameter logging")

    logger.info(f"MRI_FOLDER: {mri_folder}")
    logger.info(f"OUTPUT_DIR: {output_dir}")
    logger.info(f"CSV_PATH: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    failed_nodes =[]

    df = pd.read_csv(csv_path)
    filtered_df = df[df['Matted'] == 'matted']
    processed_fn_list = [Path(file_path).name.replace('.nii.gz', '') for file_path in filtered_df['Case']]
    matted_nodes = list(zip(processed_fn_list, filtered_df['Node'].astype(str)))

    logger.info(f"Found {len(matted_nodes)} matted nodes where Matted is 'matted'")
    logger.debug(f"matted_nodes is {matted_nodes}")

    mri_files = [os.path.join(mri_folder, f) for f in os.listdir(mri_folder) 
                if f.endswith('.nii.gz')]

    logger.info(f"Found {len(mri_files)} mri files from mri_foler path")

    for matted_node in tqdm(matted_nodes, desc="Extracting matted nodes", unit="node"):
        logger.info(f"......Starting process for matted node {matted_node}")
        try:
            target_fn_header = matted_node[0] + "_node" + matted_node[1]
            logger.info(f"target_fn_header is {target_fn_header}")
            target_mri_files = []
    
            for file in mri_files:
                if Path(file).name.startswith(target_fn_header):
                    target_mri_files.append(file)

            logger.info(f"target_mri_files (length {len(target_mri_files)}) is {target_mri_files}")

            if target_mri_files:
                for source_file in target_mri_files:
                    try:
                        # Get just the filename from the full path
                        filename = os.path.basename(source_file)
                        # Create the destination path
                        destination = os.path.join(output_dir, filename)
                        # Copy the file
                        shutil.copy2(source_file, destination)
                        logger.debug(f"Copied {source_file} to {destination}")
                    except Exception as copy_error:
                        logger.error(f"Failed to copy {source_file}: {str(copy_error)}")
                        failed_nodes.append((matted_node, f"Copy error: {str(copy_error)}"))
                
                logger.info(f"Successfully copied {len(target_mri_files)} files for node {matted_node}")
            else:
                logger.warning(f"No files found for matted node {matted_node}")

        except Exception as e:
            logger.error(f"Error processing {matted_node}: {str(e)}")
            failed_nodes.append((matted_node, str(e)))
            continue



