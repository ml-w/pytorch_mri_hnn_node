import pandas as pd
import sys
import os
import logging
import argparse

def setup_logger():
    logger = logging.getLogger(__name__)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False
    log_file = os.path.join('logs/logs_create_p_id.log')
    
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

def process_filenames_in_csv(input_path, output_path):
    """
    Reads a CSV file, extracts information from a 'filename' column,
    and saves the result to a new CSV file.

    The function creates two new columns:
    1. 'patient_id': Everything before the word 'node'.
    2. 'node_and_slice': The part starting from 'node' but with the
                        '.nii.gz' extension removed.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path where the output CSV file will be saved.
    """
    if not os.path.exists(input_path):
        logger.info(f"Error: The file '{input_path}' was not found.")

    try:
        logger.info(f"Reading data from '{input_path}'...")
        df = pd.read_csv(input_path)

        if 'filename' not in df.columns:
            logger.info(f"Error: The required column 'filename' was not found in '{input_path}'.")

        # This regex has two capturing groups:
        # Group 1: (.*?)      - Captures everything non-greedily from the start up to 'node'.
        #           node      - Matches the literal string 'node'.
        # Group 2: (.*)       - Captures the rest of the string.
        #           \.nii\.gz - Matches the literal '.nii.gz' extension.
        # The '$' at the end ensures the match happens at the end of the string.
        regex = r"^(.*?)node(.*)\.nii\.gz$"

        # .str.extract() returns a new DataFrame with one column per capture group.
        logger.info("Extracting 'patient_id' and 'node_and_slice' from filenames...")
        extracted_data = df['filename'].str.extract(regex)

        # Assign the extracted data to the new columns
        # also prepend 'node' back to the second column as the regex captured what came after it
        df['patient_id'] = extracted_data[0]
        df['node_and_slice'] = 'node' + extracted_data[1]

        logger.info(f"Saving processed data to '{output_path}'...")
        df.to_csv(output_path, index=False)
        logger.info("Processing complete.")

    except Exception as e:
        logger.info(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Process a csv file from create_labels.py and extract patient_id and node_and_slice from filenames.")
    parser.add_argument('--input_csv_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_csv_file', type=str, required=True, help='Path where the output CSV file will be saved.')

    args = parser.parse_args()

    input_csv_file = args.input_csv_file
    output_csv_file = args.output_csv_file

    logger.info(f"Starting parameter logging")

    logger.info(f"INPUT_CSV_FILE: {input_csv_file}")
    logger.info(f"OUTPUT_CSV_FILE: {output_csv_file}")

    process_filenames_in_csv(input_csv_file, output_csv_file)

    logger.debug(f"\n--- Content of '{output_csv_file}' ---")
    with open(output_csv_file, 'r') as f:
        logger.debug(f.read())