import logging
import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import sys
from pathlib import Path
from random import randint
import math
import argparse

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# MRI_FOLDER = "data/raw/images/"

# ANNOTATION_FOLDER = "output/aug5/"
# OUTPUT_DIR = "output/extract_aug5"

# ANNOTATION_FOLDER = "output/valid_lavels/"
# OUTPUT_DIR = "output/extract_real"

# MRI_FOLDER = "data/test_small/images"
# ANNOTATION_FOLDER = "data/test_small/labels"
# OUTPUT_DIR = "output/test_small"

def setup_logger():
    logger = logging.getLogger(__name__)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False
    log_file = os.path.join(log_dir, 'logs_2dx3.log')
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) 
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

SW_STRIDE = 1
IMG_PADDING = 3 # TODO: confirm unit


logger.info(f"Starting parameter logging")
logger.info(f"SW_STRIDE: {SW_STRIDE}")
logger.info(f"IMG_PADDING: {IMG_PADDING}")
logger.debug("Debug logging is enabled")

def match_files(mri_files, annotation_files):
    """Match MRI files with their corresponding annotation files based on filename."""
    pairs = []
    matched_annotation_files = set()
    
    for mri_file in mri_files:
        # Extract the base filename without path
        mri_basename = os.path.basename(mri_file)
        
        # Look for a matching annotation file
        for anno_file in annotation_files:
            if os.path.basename(anno_file) == mri_basename:
                pairs.append((mri_file, anno_file))
                matched_annotation_files.add(anno_file)
                break

    for anno_file in annotation_files:
        if anno_file not in matched_annotation_files:
            logger.info(f"No MRI file found for annotation file: {anno_file}")
    
    return pairs


class DataLoader:
    def __init__(self, mri_path, annotation_path):
        self.mri_path = mri_path
        self.annotation_path = annotation_path

        self.mri_image = None
        self.annotation_image = None
        self.mri_np = None
        self.annotation_np = None
        self.spacing = None
        self.origin = None
        self.size = None
        self.node_labels = None
        self.node_stats = {}
        self.node_masks = {}

        logger.info("DataLoader initialized")

    def load_data(self):
        """Load MRI and annotation data + some checking."""
        logger.info(f"Loading MRI image from {self.mri_path}")
        self.mri_image = sitk.ReadImage(self.mri_path)
        self.mri_np = sitk.GetArrayFromImage(self.mri_image)
        
        logger.info(f"Loading annotation image from {self.annotation_path}")
        self.annotation_image = sitk.ReadImage(self.annotation_path)
        self.annotation_np = sitk.GetArrayFromImage(self.annotation_image)

        # Ensure same coordinate system
        if not self.check_coordinate_match():
            logger.warning("MRI and annotation images might not be in the same coordinate system!")
        
        self.spacing = self.mri_image.GetSpacing()
        logger.info(f"Image spacing: {self.spacing}")

        self.origin = self.mri_image.GetOrigin()
        logger.info(f"Image origin: {self.origin}")

        self.size = self.mri_image.GetSize()
        logger.info(f"Image size: {self.size}")
        
        # Extract node labels
        np_annotation = sitk.GetArrayFromImage(self.annotation_image)
        self.node_labels = np.unique(np_annotation)
        self.node_labels = self.node_labels[self.node_labels > 0]  # Remove background
        
        logger.info(f"Found {len(self.node_labels)} lymph node annotations with labels: {self.node_labels}")
        
        # Create individual masks for each node
        self.create_node_masks()
        
        return self
        
    def check_coordinate_match(self):
        """Helper for the load_data function"""
        """Check if MRI and annotation images have matching coordinate systems."""
        mri_size = self.mri_image.GetSize()
        anno_size = self.annotation_image.GetSize()
        mri_spacing = self.mri_image.GetSpacing()
        anno_spacing = self.annotation_image.GetSpacing()
        mri_origin = self.mri_image.GetOrigin()
        anno_origin = self.annotation_image.GetOrigin()
        
        size_match = mri_size == anno_size
        spacing_match = all(abs(m - a) < 1e-3 for m, a in zip(mri_spacing, anno_spacing))
        origin_match = all(abs(m - a) < 1e-3 for m, a in zip(mri_origin, anno_origin))

        self.num_slides = anno_size[2]
        
        logger.info(f"Size match: {size_match}, Spacing match: {spacing_match}, Origin match: {origin_match}")
        logger.info(f"MRI spacing: {mri_spacing}, Anno spacing: {anno_spacing}")
        logger.info(f"xyz: {anno_size}, num_slides: {self.num_slides}")
        
        return size_match and spacing_match and origin_match
    
    def create_node_masks(self):
        """Helper for the load_data function"""
        """Create binary masks for each lymph node."""
        for label in self.node_labels:
            logger.info(f"Creating mask for node {label}")
            
            # Create binary mask for this node
            node_mask = sitk.Equal(self.annotation_image, int(label))
            self.node_masks[label] = node_mask
            
            # Calculate basic statistics for this node
            np_mri = sitk.GetArrayFromImage(self.mri_image)
            np_mask = sitk.GetArrayFromImage(node_mask)
            node_voxels = np_mri[np_mask > 0]
            
            if len(node_voxels) > 0:
                self.node_stats[label] = {
                    'mean_intensity': np.mean(node_voxels),
                    'std_intensity': np.std(node_voxels),
                    'volume_mm3': np.sum(np_mask) * np.prod(self.spacing),
                    'voxel_count': np.sum(np_mask)
                }
                logger.info(f"  Node {label} stats: {self.node_stats[label]}")
            else:
                logger.warning(f"  Node {label} has no voxels!")

    def get_slices_with_mask(self, node_label):
        """
        Returns a list of slice IDs where the specified node mask exists.
        
        Args:
            node_label: The label of the node to check
            
        Returns:
            List of slice IDs (z-indices) containing the mask
        """
        if node_label not in self.node_masks:
            logger.error(f"Node label {node_label} not found in node masks!")
            return []
        
        # Convert the SimpleITK mask to a numpy array
        mask_array = sitk.GetArrayFromImage(self.node_masks[node_label]) > 0
        
        # Find slices where the mask has at least one True value
        # The first dimension in the numpy array corresponds to the z-axis (slices)
        slices_with_mask = []
        for slice_id in range(mask_array.shape[0]):
            if np.any(mask_array[slice_id]):
                slices_with_mask.append(slice_id)
        
        logger.debug(f"Node {node_label} appears in {len(slices_with_mask)} slices: {slices_with_mask}")
        
        return slices_with_mask

def get2dx3(label, label_masks, id_list, mri_np, spacing, origin, mask_np):
    np_label_masks = sitk.GetArrayFromImage(label_masks)
    triplets = []
    list_image_stack_sitk = []
    # list_mask_stack_sitk = []

    logger.debug(f"id list is {id_list}")
    triplets = [((id_list[i]-1), id_list[i], (id_list[i]+1)) 
                for i in range(len(id_list))]
    logger.info(f"generated triplet list {triplets}")

    mri_np_depth = mri_np.shape[0]

    triplets = [
        (z1, z2, z3) for (z1, z2, z3) in triplets
        if 0 <= z1 < mri_np_depth and 0 <= z2 < mri_np_depth and 0 <= z3 < mri_np_depth
    ]

    logger.info(f"valid triplet list is {triplets}")
        
    slice_crops = {}

    for slice_id in id_list:
        np_slice_mask = np_label_masks[slice_id]
        rows, cols = np.where(np_slice_mask > 0)

        if len(rows) == 0 or len(cols) == 0:
            continue

        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        width = max_col - min_col
        height = max_row - min_row

        side = max(width, height)

        centroid_row = (min_row + max_row) // 2
        centroid_col = (min_col + max_col) // 2

        half_side = side // 2

        box_min_row = max(0, centroid_row - half_side - IMG_PADDING)
        box_max_row = min(mri_np.shape[1] - 1, centroid_row + half_side + IMG_PADDING)
        box_min_col = max(0, centroid_col - half_side - IMG_PADDING)
        box_max_col = min(mri_np.shape[2] - 1, centroid_col + half_side + IMG_PADDING)

        new_origin_x = origin[0] + (box_min_col * spacing[0]) 
        new_origin_y = origin[1] + (box_min_row * spacing[1]) 
        new_origin_z = origin[2] + (slice_id * spacing[2]) 

        slice_crops[slice_id] = {
            'new_origin': (new_origin_x, new_origin_y, new_origin_z),
            'centroid_row': centroid_row,
            'centroid_col': centroid_col,
            'half_side': half_side
        }
    logger.debug(f"slice_crops keys are {list(slice_crops.keys())}")


    for i, (z1, z2, z3) in enumerate(triplets):

        new_origin = slice_crops[z2]['new_origin']
        
        # mask_np2 = slice_crops[z2]['mask_crop']

        centroid_row = slice_crops[z2]['centroid_row']
        centroid_col = slice_crops[z2]['centroid_col']
        half_side = slice_crops[z2]['half_side']
        logger.debug(f"centroid row col half side origin is according to z2: {centroid_row} {centroid_col} {half_side} {new_origin}")

        if z1 in id_list or z3 in id_list:
            if z1 in id_list:
                if slice_crops[z1]['half_side'] > half_side:
                    centroid_row = slice_crops[z1]['centroid_row']
                    centroid_col = slice_crops[z1]['centroid_col']
                    half_side = slice_crops[z1]['half_side']
                    new_origin = slice_crops[z1]['new_origin']
                    logger.debug(f"new centroid row col half side origin is according to z1: {centroid_row} {centroid_col} {half_side} {new_origin}")
                else:
                    logger.debug(f"z1 in id_list but z1 half side not larger than current")
            if z3 in id_list:
                if slice_crops[z3]['half_side'] > half_side:
                    centroid_row = slice_crops[z3]['centroid_row']
                    centroid_col = slice_crops[z3]['centroid_col']
                    half_side = slice_crops[z3]['half_side']
                    new_origin = slice_crops[z3]['new_origin']
                    logger.debug(f"new centroid row col half side origin is according to z3: {centroid_row} {centroid_col} {half_side} {new_origin}")
                else:
                    logger.debug(f"z3 in id_list but z3 half side not larger than current")

        min_half_side_px = math.ceil(7.5 / spacing[0])
        if half_side < min_half_side_px:
            old_half_side = half_side
            half_side = min_half_side_px
            logger.debug(f"Applied minimum half_side: increased from {old_half_side} to {half_side} pixels "
                         f"({half_side * spacing[0]:.2f} mm half-side)")
            half_side = half_side - IMG_PADDING
            logger.debug(f"so img padding is removed from half_side, making half side {half_side}")

        box_min_row = max(0, centroid_row - half_side - IMG_PADDING)
        box_max_row = min(mri_np.shape[1] - 1, centroid_row + half_side + IMG_PADDING)
        box_min_col = max(0, centroid_col - half_side - IMG_PADDING)
        box_max_col = min(mri_np.shape[2] - 1, centroid_col + half_side + IMG_PADDING)

        image_np1 = mri_np[z1, box_min_row:box_max_row+1, box_min_col:box_max_col+1]
        image_np2 = mri_np[z2, box_min_row:box_max_row+1, box_min_col:box_max_col+1]
        image_np3 = mri_np[z3, box_min_row:box_max_row+1, box_min_col:box_max_col+1]
        # mask_np1 = mask_np[z1, box_min_row:box_max_row+1, box_min_col:box_max_col+1]
        # mask_np3 = mask_np[z3, box_min_row:box_max_row+1, box_min_col:box_max_col+1]

        image_stack = np.stack([image_np1, image_np2, image_np3], axis=0)
        # mask_stack = np.stack([mask_np1, mask_np2, mask_np3], axis=0)

        image_stack_sitk = sitk.GetImageFromArray(image_stack)
        image_stack_sitk.SetSpacing(spacing)
        image_stack_sitk.SetOrigin(new_origin)
        # mask_stack_sitk = sitk.GetImageFromArray(mask_stack)
        # mask_stack_sitk.SetSpacing(spacing)
        # mask_stack_sitk.SetOrigin(new_origin)

        logger.info(f"generated sitk stack for {i+1}/{len(triplets)}, z123 is {(z1, z2, z3)}")

        list_image_stack_sitk.append(image_stack_sitk)
        # list_mask_stack_sitk.append(mask_stack_sitk)
        
    
    return list_image_stack_sitk

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process MRI and annotation folders.")
    parser.add_argument('--mri_folder', type=str, required=True, help='Path to the MRI folder')
    parser.add_argument('--annotation_folder', type=str, required=True, help='Path to the annotation folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')

    args = parser.parse_args()

    mri_folder = args.mri_folder
    annotation_folder = args.annotation_folder
    output_dir = args.output_dir

    logger.info(f"MRI_FOLDER: {mri_folder}")
    logger.info(f"ANNOTATION_FOLDER: {annotation_folder}")
    logger.info(f"OUTPUT_DIR: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    IMAGES_DIR = os.path.join(output_dir, "images")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    LABELS_DIR = os.path.join(output_dir, "labels")
    os.makedirs(LABELS_DIR, exist_ok=True)
    
    # Get all files in both folders
    mri_files = [os.path.join(mri_folder, f) for f in os.listdir(mri_folder) 
                if f.endswith('.nii.gz')]
    
    annotation_files = [os.path.join(annotation_folder, f) for f in os.listdir(annotation_folder) 
                       if f.endswith('.nii.gz')]
    
    # Match MRI files with corresponding annotation files
    file_pairs = match_files(mri_files, annotation_files)

    logger.info(f"file pairs are {file_pairs}")

    logger.info(f"Found {len(file_pairs)} matching pairs out of {len(mri_files)} MRI files and {len(annotation_files)} annotation files")

    for mri_path, annotation_path in tqdm(file_pairs, desc="Processing file pairs", unit="pair"):
        logger.info(f"............Starting process for {mri_path} and {annotation_path}")
        subj_id = Path(mri_path).stem.split('.')[0]
        try:
            dataloader = DataLoader(mri_path, annotation_path)
            dataloader.load_data();
            node_labels = dataloader.node_labels
            logger.info(f"retrieved node_labels, which is {node_labels}")
            node_masks = dataloader.node_masks
            node_masks_np = dataloader.annotation_np
            mri_np = dataloader.mri_np
            mri_image = dataloader.mri_image
            size = dataloader.size
            spacing = dataloader.spacing
            origin = dataloader.origin

            for label in node_labels:
                logger.info(f"processing node {label}")
                label_masks = node_masks[label]
                logger.info(f"retrieved label_masks, length is {len(label_masks)}")
                id_list = dataloader.get_slices_with_mask(label)
                logger.info(f"retrieved id_list, length is {len(id_list)}, this node appears in {id_list}")
            
                list_image_stack_sitk = get2dx3(label, label_masks, id_list, mri_np, spacing, origin, node_masks_np)
                
                if list_image_stack_sitk:
                    i = 0
                    for i in range(len(list_image_stack_sitk)):
                        output_filename = f"{os.path.basename(subj_id)}_node{label}_2dx3_{i}.nii.gz"
                        output_path = os.path.join(IMAGES_DIR, output_filename)

                        sitk.WriteImage(list_image_stack_sitk[i], output_path)

                        logger.info(f"saved file with filename {output_filename}")
                        logger.info(f"it has size: {list_image_stack_sitk[i].GetSize()} and spacing {list_image_stack_sitk[i].GetSpacing()} and origin {list_image_stack_sitk[i].GetOrigin()}")

                    # i = 0
                    # for i in range(len(list_mask_stack_sitk)):
                    #     output_filename = f"mask_{os.path.basename(subj_id)}_node{label}_2dx3_{i}.nii.gz"
                    #     output_path = os.path.join(LABELS_DIR, output_filename)

                    #     sitk.WriteImage(list_mask_stack_sitk[i], output_path)

                    #     logger.info(f"saved file with filename {output_filename}")
                    #     logger.info(f"it has size: {list_mask_stack_sitk[i].GetSize()} and spacing {list_mask_stack_sitk[i].GetSpacing()} and origin {list_mask_stack_sitk[i].GetOrigin()}")
                else:
                    logger.info(f"no images can be extracted")

        except Exception as e:
            logger.error(f"Error processing {mri_path}: {str(e)}")
            continue