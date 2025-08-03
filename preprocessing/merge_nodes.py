import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import ndimage
import logging
import datetime
import pandas as pd
import sys
import os
from tqdm import tqdm

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Constants
DISTANCE_THRESHOLD = 2.0 # mm
DISTANCE_THRESHOLD = 4.0 # higher threshold used to augment data
CONTACT_AREA_THRESHOLD_RATIO = 0.1 # relative threshold
CONTACT_AREA_THRESHOLD_RATIO = 0.01 # lower threshold used to augment data
# CONTACT_AREA_THRESHOLD_RATIO = 0.01 # relative threshold, FOR DEBUGGING PURPOSES, PLS USE THE ABOVE ONE
INTENSITY_DIFF_THRESHOLD = 0.2 # relative threshold
INTENSITY_DIFF_THRESHOLD = 0.99 # higher threshold used to augment data
# INTENSITY_DIFF_THRESHOLD = 0.5 # relative threshold, FOR DEBUGGING PURPOSES, PLS USE THE ABOVE ONE
DILATION_RADIUS = 1 # voxels
DILATION_RADIUS = 8 # voxels, used to augment data
# DILATION_RADIUS = 3 # voxels, FOR DEBUGGING PURPOSES, PLS USE THE ABOVE ONE
DEBUG = False # for later functions
MRI_FOLDER = "data/raw/images/"
ANNOTATION_FOLDER = "output/valid_labels/"
OUTPUT_DIR = "output/aug5"
MAJ_VOTE_THRES = 0.25 # threshold to pass majority vote

def setup_logger():
    logger = logging.getLogger(__name__)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    logger.setLevel(logging.DEBUG)
    
    logger.propagate = False
    log_file = os.path.join(log_dir, 'logs.log')
    
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

# Log parameters once
logger.info(f"Starting parameter logging")
logger.info(f"DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}")
logger.info(f"CONTACT_AREA_THRESHOLD_RATIO: {CONTACT_AREA_THRESHOLD_RATIO}")
logger.info(f"INTENSITY_DIFF_THRESHOLD: {INTENSITY_DIFF_THRESHOLD}")
logger.info(f"DILATION_RADIUS: {DILATION_RADIUS}")
logger.info(f"MRI_FOLDER: {MRI_FOLDER}")
logger.info(f"ANNOTATION_FOLDER: {ANNOTATION_FOLDER}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"MAJ_VOTE_THRES: {MAJ_VOTE_THRES}")
logger.debug("Debug logging is enabled")

class DataLoader:
    def __init__(self, mri_path, annotation_path):
        self.mri_path = mri_path
        self.annotation_path = annotation_path

        self.mri_image = None
        self.annotation_image = None
        self.spacing = None
        self.num_slides = None
        self.node_labels = None
        self.node_masks = {}
        self.node_stats = {}
        self.adjacency_graph = None

        logger.info("DataLoader initialized")

    def load_data(self):
        """Load MRI and annotation data + some checking."""
        logger.info(f"Loading MRI image from {self.mri_path}")
        self.mri_image = sitk.ReadImage(self.mri_path)
        
        logger.info(f"Loading annotation image from {self.annotation_path}")
        self.annotation_image = sitk.ReadImage(self.annotation_path)

        # Ensure same coordinate system
        if not self.check_coordinate_match():
            logger.warning("MRI and annotation images might not be in the same coordinate system!")
        
        self.spacing = self.mri_image.GetSpacing()
        logger.info(f"Image spacing: {self.spacing}")
        
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

    def get_common_slices(self, node_a, node_b):
        """
        Returns slice IDs where both node masks exist.
        
        Args:
            node_a: First node label
            node_b: Second node label
            
        Returns:
            List of slice IDs where both masks are present
        """
        slices_a = set(self.get_slices_with_mask(node_a))
        slices_b = set(self.get_slices_with_mask(node_b))
        
        common_slices = sorted(list(slices_a.intersection(slices_b)))
        
        logger.info(f"Nodes {node_a} and {node_b} appear together in {len(common_slices)} slices: {common_slices}")
        
        return common_slices

class SliceAnalyzer:
    # def __init__(self, node_a, node_b, slice_id):
    #     self.node_a = node_a
    #     self.node_b = node_b
    #     self.slice_id = slice_id

    def __init__(self, node_masks, spacing):
        self.node_masks = node_masks;
        self.spacing = spacing;
        logger.info("DataLoader initialized")

    # Criteria 1: Minimum distance between the two nodes in this slice        
    def calculate_min_distance_single_slice(self, node_a, node_b, slice_id, debug=False):
        """
        Calculate minimum distance between two nodes in a single specified slice.
        
        Args:
            node_a: First node identifier
            node_b: Second node identifier
            slice_id: The specific slice to analyze
            
        Returns:
            Minimum distance between the two nodes in the specified slice
            and a visualization for debugging
        """
        # Get the 3D masks
        mask_a_3d = sitk.GetArrayFromImage(self.node_masks[node_a]) > 0
        mask_b_3d = sitk.GetArrayFromImage(self.node_masks[node_b]) > 0
        
        # Extract only the specified slice
        if slice_id < 0 or slice_id >= mask_a_3d.shape[0]:
            logger.error(f"Slice ID {slice_id} out of range (0-{mask_a_3d.shape[0]-1})")
            return np.inf, None
        
        # Extract the 2D masks for the specified slice
        mask_a = mask_a_3d[slice_id]
        mask_b = mask_b_3d[slice_id]
        
        # If either mask is empty in this slice, return infinity
        if not np.any(mask_a) or not np.any(mask_b):
            logger.warn(f"One or both masks are empty in slice {slice_id}")
            return np.inf, None
        
        # Get coordinates of boundary pixels
        # A pixel is on the boundary if it's part of the mask and has at least one neighbor that isn't
        struct = ndimage.generate_binary_structure(2, 1)  # 2D connectivity
        eroded_a = ndimage.binary_erosion(mask_a, struct)
        boundary_a = mask_a & ~eroded_a
        
        eroded_b = ndimage.binary_erosion(mask_b, struct)
        boundary_b = mask_b & ~eroded_b
        
        # Get indices of boundary pixels
        boundary_a_indices = np.argwhere(boundary_a)
        boundary_b_indices = np.argwhere(boundary_b)
        
        # Convert indices to physical coordinates using spacing
        # Using only the x,y components of spacing for 2D
        spacing_xy = self.spacing[0:2]
        boundary_a_coords = boundary_a_indices * spacing_xy
        boundary_b_coords = boundary_b_indices * spacing_xy
        logger.debug(f"Spacing being used: {self.spacing}")
        logger.debug(f"Spacing_xy: {spacing_xy}")
        
        # Calculate minimum distance using KDTree for efficiency
        from scipy.spatial import KDTree
        
        if len(boundary_a_coords) == 0 or len(boundary_b_coords) == 0:
            logger.warning(f"One or both boundaries are empty in slice {slice_id}")
            return np.inf, None
        
        tree_a = KDTree(boundary_a_coords)
        tree_b = KDTree(boundary_b_coords)
        
        # Find minimum distance from A to B and get the closest points
        distances_a_to_b, indices_a_to_b = tree_a.query(boundary_b_coords)
        min_dist_a_to_b = np.min(distances_a_to_b)
        min_idx_a_to_b = indices_a_to_b[np.argmin(distances_a_to_b)]
        closest_point_a = boundary_a_coords[min_idx_a_to_b]
        closest_point_b_from_a = boundary_b_coords[np.argmin(distances_a_to_b)]
        
        # Find minimum distance from B to A and get the closest points
        distances_b_to_a, indices_b_to_a = tree_b.query(boundary_a_coords)
        min_dist_b_to_a = np.min(distances_b_to_a)
        min_idx_b_to_a = indices_b_to_a[np.argmin(distances_b_to_a)]
        closest_point_b = boundary_b_coords[min_idx_b_to_a]
        closest_point_a_from_b = boundary_a_coords[np.argmin(distances_b_to_a)]
        
        # Determine which is the minimum distance
        if min_dist_a_to_b <= min_dist_b_to_a:
            min_dist = min_dist_a_to_b
            closest_pair = (closest_point_a, closest_point_b_from_a)
        else:
            min_dist = min_dist_b_to_a
            closest_pair = (closest_point_a_from_b, closest_point_b)
        
        if debug:
            # Create visualization for debugging
            visualization = self.create_distance_visualization(
                mask_a, mask_b, boundary_a, boundary_b, 
                closest_pair, min_dist, slice_id, node_a, node_b
            )
        
        return min_dist
    
    def create_distance_visualization(self, mask_a, mask_b, boundary_a, boundary_b, 
                                    closest_pair, min_dist, slice_id, node_a, node_b):
        """
        Create a visualization image for debugging the distance calculation.
        
        Args:
            mask_a, mask_b: Binary masks for the two nodes
            boundary_a, boundary_b: Binary masks for the boundaries
            closest_pair: Tuple of coordinates for the closest points
            min_dist: The calculated minimum distance
            slice_id: The slice being visualized
            node_a, node_b: Node identifiers
            
        Returns:
            A matplotlib figure object with the visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import ConnectionPatch
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a combined image for visualization - RGB only (no alpha channel)
        vis_img = np.zeros((*mask_a.shape, 3), dtype=float)
        
        # Fill with original masks (using semi-transparent colors)
        vis_img[mask_a, 0] = 0.7  # Red component for mask A
        vis_img[mask_b, 2] = 0.7  # Blue component for mask B
        
        # Highlight the boundaries
        vis_img[boundary_a, 0] = 1.0  # Bright red for boundary A
        vis_img[boundary_b, 2] = 1.0  # Bright blue for boundary B
        
        # Display the image
        ax.imshow(vis_img)
        
        # Add the connection line between the closest points
        if closest_pair:
            point_a, point_b = closest_pair
            # Convert from physical coordinates back to pixel indices
            spacing_xy = self.spacing[0:2]
            idx_a = point_a / spacing_xy
            idx_b = point_b / spacing_xy
            
            # Draw a line connecting the closest points
            ax.add_patch(ConnectionPatch(
                xyA=(idx_a[1], idx_a[0]),
                xyB=(idx_b[1], idx_b[0]),
                coordsA="data", coordsB="data",
                axesA=ax, axesB=ax,
                color="yellow", linewidth=2
            ))
            
            # Mark the points
            ax.plot(idx_a[1], idx_a[0], 'o', color='green', markersize=8)
            ax.plot(idx_b[1], idx_b[0], 'o', color='green', markersize=8)
            
        # Add labels and title
        ax.set_title(f"Distance between nodes {node_a} and {node_b} in slice {slice_id}: {min_dist:.2f} units")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.5, label=f'Node {node_a}'),
            Patch(facecolor='blue', alpha=0.5, label=f'Node {node_b}'),
            Patch(facecolor='yellow', label='Minimum distance')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        return fig

    def dilate_mask(self, mask: sitk.Image) -> sitk.Image:
        """
        Dilate a binary mask with SimpleITK .
        The operation is applied to every axial (x–y) slice individually.

        Parameters
        mask : sitk.Image
            3-D binary image (0 background, >0 foreground).

        Returns
        sitk.Image
            Dilated 3-D mask (uint8, 0/1) with the same meta-data as the input.
        """
        # 1. Ensure the mask is strictly 0/1
        original_mask = mask
        binary_mask   = sitk.Cast(mask > 0, sitk.sitkUInt8)

        original_count = int(sitk.GetArrayViewFromImage(binary_mask).sum())

        # 2. Prepare the 2-D extractor and dilater
        size  = list(binary_mask.GetSize()) # [x, y, z]
        depth = size[2]

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], 0])

        dilater = sitk.BinaryDilateImageFilter()
        dilater.SetForegroundValue(1)
        dilater.SetBackgroundValue(0)
        dilater.SetKernelType(sitk.sitkBall)
        dilater.SetKernelRadius(DILATION_RADIUS)

        # 3. Dilate every slice and collect the results
        dilated_slices = []
        for z in range(depth):
            extractor.SetIndex([0, 0, z])
            slice2d        = extractor.Execute(binary_mask)
            dilated_slice  = dilater.Execute(slice2d)
            dilated_slices.append(dilated_slice)

        # 4. Stack the 2-D slices back into a 3-D volume
        dilated_volume = sitk.JoinSeries(dilated_slices)
        # Restoring the original meta-data
        dilated_volume.CopyInformation(original_mask)

        # 5. Logging
        dilated_count = int(sitk.GetArrayViewFromImage(dilated_volume).sum())
        logger.info(
            f"  Dilation: {original_count} voxels -> {dilated_count} voxels "
            f"(+{dilated_count - original_count}, "
            f"{dilated_count / max(original_count, 1):.2f}x)"
        )

        return dilated_volume
    
    def find_contact_region(self, dilated_a, dilated_b, node_a, node_b, debug=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        contact_region = sitk.And(dilated_a, dilated_b)
        if debug:
            filename_ab_cont = f"{timestamp}_node_{node_a}_{node_b}_cont.nii.gz"
            sitk.WriteImage(contact_region, filename_ab_cont)
            logger.info(f"Saved {filename_ab_cont}")

        return contact_region

    
    # Criteria 2: After dilation, area of overlapping region in this slice
    def calculate_contact_area(self, contact_region_slice):
        """Note that contact_region_slice should be a single layer (2D not 3D)"""

        np_contact = sitk.GetArrayFromImage(contact_region_slice)
        spacing_xy = self.spacing[0:2]
        voxel_area = np.prod(spacing_xy)
        contact_voxels = np.sum(np_contact)
        area = contact_voxels * voxel_area

        logger.debug(f"Spacing xy: {spacing_xy}")
        logger.debug(f"Voxel area: {voxel_area}")
        logger.debug(f"Contact region: {contact_voxels} voxels")
        logger.debug(f"Contact area: {area} mm2")

        return area
        
    # Criteria 3: After dilation, intensity of overlapping region in this slice, relative to intensity of each of the two nodes   
    def calculate_intensity_similarity(self, np_mri_slice, original_a_slice, original_b_slice, contact_region_slice):
        np_original_a_slice = sitk.GetArrayFromImage(original_a_slice)
        np_original_b_slice = sitk.GetArrayFromImage(original_b_slice)
        np_contact = sitk.GetArrayFromImage(contact_region_slice)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        ##### Debugging: checked that the images and the masks do line up
        # np_mri_check = sitk.GetImageFromArray(np_mri_slice)
        # np_original_a_slice_check = sitk.GetImageFromArray(np_original_a_slice)
        # np_original_b_slice_check = sitk.GetImageFromArray(np_original_b_slice)
        # np_contact_check = sitk.GetImageFromArray(np_contact)

        # fn_np_mri_check = f"{timestamp}_np_mri_check.nii.gz"
        # fn_np_original_a_slice_check = f"{timestamp}_np_original_a_slice_check.nii.gz"
        # fn_np_original_b_slice_check = f"{timestamp}_np_original_b_slice_check.nii.gz"
        # fn_np_contact_check = f"{timestamp}_contact_check.nii.gz"

        # sitk.WriteImage(np_mri_check, fn_np_mri_check)
        # sitk.WriteImage(np_original_a_slice_check, fn_np_original_a_slice_check)
        # sitk.WriteImage(np_original_b_slice_check, fn_np_original_b_slice_check)
        # sitk.WriteImage(np_contact_check, fn_np_contact_check)

        if np.sum(np_contact) == 0:
            logger.warning("Contact region is empty")
            return False, 0
        
        if np.sum(np_original_a_slice) == 0:
            logger.warning("Original a slice is empty")
            return False, 0
        
        if np.sum(np_original_b_slice) == 0:
            logger.warning("Original b slice is empty")
            return False, 0
        
        ori_a_intensities = np_mri_slice[np_original_a_slice > 0]
        mean_ori_a = np.mean(ori_a_intensities)
        count_ori_a = np.sum(original_a_slice)

        ori_b_intensities = np_mri_slice[np_original_b_slice > 0]
        mean_ori_b = np.mean(ori_b_intensities)
        count_ori_b = np.sum(original_b_slice)

        mean_ori_w = (mean_ori_a * count_ori_a + mean_ori_b * count_ori_b) / (count_ori_a + count_ori_b)

        contact_intensities = np_mri_slice[np_contact > 0]
        mean_contact = np.mean(contact_intensities)

        logger.debug(f"mean_ori_a: {mean_ori_a}, count_ori_a: {count_ori_a}, mean_ori_b: {mean_ori_b}, count_ori_b: {count_ori_b}")
        logger.debug(f"mean_ori_w: {mean_ori_w}, mean_contact: {mean_contact}")

        rel_diff = abs(mean_contact - mean_ori_w) / mean_ori_w
        
        is_similar = rel_diff <= INTENSITY_DIFF_THRESHOLD

        logger.info(f"Relative difference is {rel_diff}")

        return is_similar, 1 - rel_diff

def run_pipeline_on_case(mri_path, annotation_path, debug=False):
    dataloader = DataLoader(mri_path=mri_path, annotation_path=annotation_path)
    dataloader.load_data();

    node_labels = dataloader.node_labels
    adjacency_graph = nx.Graph()
    for label in node_labels:
        adjacency_graph.add_node(label)

    node_pairs = [(a, b) for i, a in enumerate(node_labels) 
                     for b in node_labels[i+1:]]
        
    logger.info(f"Analyzing {len(node_pairs)} node pairs")

    node_masks = dataloader.node_masks
    spacing = dataloader.spacing
    spacing_xy = spacing[0:2]
    voxel_area = np.prod(spacing_xy)
    sliceanalyzer = SliceAnalyzer(node_masks=node_masks, spacing=spacing);

    mri_image = dataloader.mri_image
    np_mri = sitk.GetArrayFromImage(mri_image)

    node_pairs_to_merge = []
    node_pairs_man_review = []
    
    for node_a, node_b in node_pairs:
        logger.info(f"Analyzing node pair ({node_a}, {node_b})")
        common_list = dataloader.get_common_slices(node_a=node_a, node_b=node_b)

        if common_list:

            # Initialize variables
            num_mat_slices = 0
            len_common_list = len(common_list)
            index_list = [f"{node_a} and {node_b}"] * len_common_list
            slide_id_list = common_list
            c1_list = [False] * len_common_list
            ful_c1 = False
            c2_list = [False] * len_common_list
            ful_c2 = False
            c3_list = [False] * len_common_list
            ful_c3 = False

            logger.info(f"Analyzing node pair ({node_a}, {node_b}) since they have slides in common")

            ############################################################## Criteria 1 ##############################################################
            logger.info(f"Starting analysis of criteria 1 for node pair ({node_a}, {node_b})")
            for slice_id in common_list:
                index_slice_id = common_list.index(slice_id)
                logger.info(f"Analyzing criteria 1 for node pair ({node_a}, {node_b}) in slice {slice_id}")
                logger.debug(f"Index of this slice in the common list is {index_slice_id}")

                min_dist = sliceanalyzer.calculate_min_distance_single_slice(node_a=node_a, node_b=node_b, slice_id=slice_id)
                logger.info(f"Minimum distance is {min_dist}")
                
                if min_dist < DISTANCE_THRESHOLD:
                    logger.debug(f"Minimum distance lower than threshold {DISTANCE_THRESHOLD}")
                    c1_list[index_slice_id] = True
                else: 
                    logger.debug(f"Minimum distance not lower than threshold {DISTANCE_THRESHOLD}")
            
            logger.info(f"In node pair ({node_a}, {node_b}), number of slices that fulfilled criteria 1 is {sum(c1_list)} out of {len_common_list}")

            if sum(c1_list) >= (len_common_list * MAJ_VOTE_THRES):
                logger.info(f"In node pair ({node_a}, {node_b}), half or more than half of total slices fulfilled criteria 1, proceeding to criteria 2 analysis")
                ful_c1 = True 
            else:
                logger.info(f"In node pair ({node_a}, {node_b}), less than {len_common_list * MAJ_VOTE_THRES} of total slices fulfilled criteria 1, skipping further analysis")
                 
            ############################################################## Criteria 2 ##############################################################
            if ful_c1:
                logger.info(f"Starting analysis of criteria 2 for node pair ({node_a}, {node_b})")
                logger.info(f"Dilating masks of node pair ({node_a}, {node_b}) with dilation radius {DILATION_RADIUS}")

                # Get original masks
                original_a = node_masks[node_a]
                original_b = node_masks[node_b]
                
                # Dilate both masks
                dilated_a = sliceanalyzer.dilate_mask(original_a)
                dilated_b = sliceanalyzer.dilate_mask(original_b)

                if debug:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                    filename_a_orig = f"{timestamp}_node_{node_a}_original.nii.gz"
                    filename_a_dil = f"{timestamp}_node_{node_a}_dilated.nii.gz"
                    
                    sitk.WriteImage(original_a, filename_a_orig)
                    logger.info(f"Saved {filename_a_orig}")
                    sitk.WriteImage(dilated_a, filename_a_dil)
                    logger.info(f"Saved {filename_a_dil}")

                    filename_b_orig = f"{timestamp}_node_{node_b}_original.nii.gz"
                    filename_b_dil = f"{timestamp}_node_{node_b}_dilated.nii.gz"
                    
                    sitk.WriteImage(original_b, filename_b_orig)
                    logger.info(f"Saved {filename_b_orig}")
                    sitk.WriteImage(dilated_b, filename_b_dil)
                    logger.info(f"Saved {filename_b_dil}")

                for slice_id in common_list:
                    index_slice_id = common_list.index(slice_id)
                    logger.info(f"Analyzing criteria 2 for node pair ({node_a}, {node_b}) in slice {slice_id}")
                    logger.debug(f"Index of this slice in the common list is {index_slice_id}")
                    
                    contact_region = sliceanalyzer.find_contact_region(dilated_a=dilated_a, dilated_b=dilated_b, node_a=node_a, node_b=node_b, debug=DEBUG)
                    
                    size_c = list(contact_region.GetSize())
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize([size_c[0], size_c[1], 0]) # Sets the size of the extracted thing
                    z = slice_id
                    extractor.SetIndex([0, 0, z]) # Sets where to start extracting from
                    contact_region_slice = extractor.Execute(contact_region)

                    size_a_ori = list(original_a.GetSize())
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize([size_a_ori[0], size_a_ori[1], 0]) # Sets the size of the extracted thing
                    z = slice_id
                    extractor.SetIndex([0, 0, z]) # Sets where to start extracting from
                    original_a_slice = extractor.Execute(original_a)
                    array_view_ori_a = sitk.GetArrayFromImage(original_a_slice)
                    pixel_count_a = int(np.sum(array_view_ori_a > 0))
                    logger.debug(f"Pixel count for node {node_a} in slice {slice_id} is {pixel_count_a} pixels")

                    size_b_ori = list(original_b.GetSize())
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize([size_b_ori[0], size_b_ori[1], 0]) # Sets the size of the extracted thing
                    z = slice_id
                    extractor.SetIndex([0, 0, z]) # Sets where to start extracting from
                    original_b_slice = extractor.Execute(original_b)
                    array_view_ori_b = sitk.GetArrayFromImage(original_b_slice)
                    pixel_count_b = int(np.sum(array_view_ori_b > 0))
                    logger.debug(f"Pixel count for node {node_b} in slice {slice_id} is {pixel_count_a} pixels")

                    pixel_count_min = min(pixel_count_a, pixel_count_b)

                    logger.info(f"Pixel count of the smaller node is {pixel_count_min}, for node pair ({node_a}, {node_b}) in slice {slice_id}")
                    
                    contact_area_threshold = pixel_count_min * CONTACT_AREA_THRESHOLD_RATIO * voxel_area

                    logger.info(f"Contact area threshold is {contact_area_threshold}, for node pair ({node_a}, {node_b}) in slice {slice_id}")

                    contact_area = sliceanalyzer.calculate_contact_area(contact_region_slice=contact_region_slice)

                    logger.info(f"Contact area of ({node_a}, {node_b}) in slice {slice_id} is {contact_area} mm2")

                    if contact_area > contact_area_threshold:
                        logger.debug(f"Contact area {contact_area} higher than threshold {contact_area_threshold}")
                        c2_list[index_slice_id] = True
                    else: 
                        logger.debug(f"Contact area {contact_area} not higher than threshold {contact_area_threshold}")
                
                logger.info(f"In node pair ({node_a}, {node_b}), number of slices that fulfilled criteria 2 is {sum(c2_list)} out of {len_common_list}")

                if sum(c2_list) >= (len_common_list * MAJ_VOTE_THRES):
                    logger.info(f"In node pair ({node_a}, {node_b}), half or more than half of total slices fulfilled criteria 2, proceeding to criteria 3 analysis")
                    ful_c2 = True 
                else:
                    logger.info(f"In node pair ({node_a}, {node_b}), less than {len_common_list * MAJ_VOTE_THRES} of total slices fulfilled criteria 2, skipping further analysis")
            ############################################################## Criteria 3 ##############################################################
            if ful_c2:
                logger.info(f"Starting analysis of criteria 3 for node pair ({node_a}, {node_b})")
                for slice_id in common_list:
                    index_slice_id = common_list.index(slice_id)
                    logger.info(f"Analyzing criteria 3 for node pair ({node_a}, {node_b}) in slice {slice_id}")
                    logger.debug(f"Index of this slice in the common list is {index_slice_id}")
                    
                    contact_region = sliceanalyzer.find_contact_region(dilated_a=dilated_a, dilated_b=dilated_b, node_a=node_a, node_b=node_b, debug=DEBUG)
                    
                    z = slice_id

                    size_c = list(contact_region.GetSize())
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize([size_c[0], size_c[1], 0]) # Sets the size of the extracted thing
                    extractor.SetIndex([0, 0, z]) # Sets where to start extracting from
                    contact_region_slice = extractor.Execute(contact_region)

                    size_a_ori = list(original_a.GetSize())
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize([size_a_ori[0], size_a_ori[1], 0]) # Sets the size of the extracted thing
                    extractor.SetIndex([0, 0, z]) # Sets where to start extracting from
                    original_a_slice = extractor.Execute(original_a)

                    size_b_ori = list(original_b.GetSize())
                    extractor = sitk.ExtractImageFilter()
                    extractor.SetSize([size_b_ori[0], size_b_ori[1], 0]) # Sets the size of the extracted thing
                    extractor.SetIndex([0, 0, z]) # Sets where to start extracting from
                    original_b_slice = extractor.Execute(original_b)

                    np_mri_slice = np_mri[z, :, :]

                    is_inten_similar, rel_simi = sliceanalyzer.calculate_intensity_similarity(np_mri_slice=np_mri_slice, original_a_slice=original_a_slice, original_b_slice=original_b_slice, contact_region_slice=contact_region_slice)
                    logger.info(f"Relative intensity similarity between contact region and ({node_a}, {node_b}) in slice {slice_id} is {rel_simi}")
                    
                    if is_inten_similar:
                        logger.debug(f"Intensity is similar according to the threshold {INTENSITY_DIFF_THRESHOLD}")
                        c3_list[index_slice_id] = True
                    else:
                        logger.debug(f"Intensity is not similar according to the threshold {INTENSITY_DIFF_THRESHOLD}")

                logger.info(f"In node pair ({node_a}, {node_b}), number of slices that fulfilled criteria 3 is {sum(c3_list)} out of {len_common_list}")

                if sum(c2_list) >= (len_common_list * MAJ_VOTE_THRES):
                    logger.info(f"In node pair ({node_a}, {node_b}), half or more than half of total slices fulfilled criteria 3, proceeding to criteria 123 analysis")
                    ful_c3 = True 
                else:
                    logger.info(f"In node pair ({node_a}, {node_b}), less than {len_common_list * MAJ_VOTE_THRES} of total slices fulfilled criteria 3, skipping further analysis")
        
            ############################################################## Criteria 123 ##############################################################
            if ful_c1 and ful_c2 and ful_c3:
                logger.info(f"Starting criteria 123 analysis for node pair ({node_a}, {node_b})")
                logger.info(f"Common list for this pair is {common_list}")
                logger.info(f"c1_list is {c1_list}")
                logger.info(f"c2_list is {c2_list}")
                logger.info(f"c3_list is {c3_list}")   

                c123_list = []

                if not (len(c1_list) == len(c2_list) == len(c3_list) == len_common_list):
                    print(f"Error: Boolean lists have different lengths: {len(c1_list)}, {len(c2_list)}, {len(c3_list)}, len_common_list: {len_common_list}")
                    return None
                
                for i in range(len_common_list):
                    c123_list.append(c1_list[i] and c2_list[i] and c3_list[i])

                logger.info(f"c123_list is {c123_list}")
                num_mat_slices = sum(c123_list)
                prop_mat_slices = num_mat_slices / len_common_list

                logger.info(f"Number of matted slices is {num_mat_slices}, number of common slices is {len_common_list}")
                logger.info(f"Proportion of matted slices is {prop_mat_slices}")

                if prop_mat_slices == MAJ_VOTE_THRES:
                    node_pairs_man_review.append((node_a, node_b))
                    logger.info(f"Manual review needed for node pair ({node_a}, {node_b})")
                elif prop_mat_slices > MAJ_VOTE_THRES:
                    node_pairs_to_merge.append((node_a, node_b))
                    logger.info(f"Added node pair ({node_a}, {node_b}) to to merge list")
                    logger.info(f"To merge list is now {node_pairs_to_merge}")
                    adjacency_graph.add_edge(node_a, node_b)
                    logger.info(f"Added edge between nodes {node_a} and {node_b} in graph")
                else:
                    logger.info(f"No need to merge node pair ({node_a}, {node_b})")
            
    
    int_node_pairs_to_merge = [(int(a), int(b)) for a, b in node_pairs_to_merge]

    logger.info(f"To merge list is {int_node_pairs_to_merge} ({node_pairs_to_merge})")

    return adjacency_graph, node_pairs_man_review, node_labels

def find_node_groups(adjacency_graph):
    nodes_to_merge = list(nx.connected_components(adjacency_graph))

    logger.info(f"Found {len(nodes_to_merge)} node / node groups:")
    for i, component in enumerate(nodes_to_merge):
        logger.info(f"  Group {i+1}: {component}")
    
    return nodes_to_merge

def merge_annotations(nodes_to_merge, annotation_path, len_node_labels):
    logger.info(f"Loading annotation image from {annotation_path}")
    annotation_image = sitk.ReadImage(annotation_path)

    merged_annotation = sitk.Cast(annotation_image, annotation_image.GetPixelID())

    min_matted_list = [False] * len_node_labels

    matted_list = [False] * len_node_labels

    for i, group in enumerate(nodes_to_merge):
        if len(group) <=1:
            logger.info(f"Skipping group {i+1} as it contains only one node")
            continue

        logger.info(f"Merging group {i+1}: {group}")

        min_matted_list[(min(group)-1)] = True

        logger.info(f"min_matted_list is now {min_matted_list}")

        for x in group:
            matted_list[(x-1)] = True

        logger.info(f"matted_list is now {matted_list}")

        new_label = min(group)

        group_mask = sitk.Image(annotation_image.GetSize(), sitk.sitkUInt8)
        group_mask.CopyInformation(annotation_image)

        # Union all node masks in this group
        for node_label in group:
            if node_label != new_label:  # Skip the new label as it will stay the same
                # Create a binary mask for this node
                temp_mask = sitk.Equal(annotation_image, int(node_label))
                
                # Add to group mask
                group_mask = sitk.Or(group_mask, temp_mask)
                
                # Remove the original node from the merged annotation by setting it to 0
                # This is equivalent to: merged_annotation = sitk.Where(temp_mask, 0, merged_annotation)
                zero_image = sitk.Image(merged_annotation.GetSize(), merged_annotation.GetPixelID())
                zero_image.CopyInformation(merged_annotation)
                
                # Multiply inverted mask with merged annotation (sets masked areas to 0)
                inverted_mask = sitk.Not(temp_mask)
                merged_annotation = sitk.Multiply(
                    merged_annotation, 
                    sitk.Cast(inverted_mask, merged_annotation.GetPixelID())
                )
        
        # Add the new label to the group areas
        # First, create an image filled with the new label
        label_image = sitk.Image(merged_annotation.GetSize(), merged_annotation.GetPixelID())
        label_image.CopyInformation(merged_annotation)
        label_image = sitk.Add(label_image, float(new_label))
        
        # Then, use masking to combine: (mask * label_image) + ((1-mask) * merged_annotation)
        merged_annotation = sitk.Add(
            sitk.Multiply(
                sitk.Cast(group_mask, merged_annotation.GetPixelID()),
                label_image
            ),
            sitk.Multiply(
                sitk.Cast(sitk.Not(group_mask), merged_annotation.GetPixelID()),
                merged_annotation
            )
        )

    logger.info("Annotation merging completed")
    return merged_annotation, min_matted_list, matted_list

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

# check for incorrect file names of annotation files first
mri_folder = MRI_FOLDER
annotation_folder = ANNOTATION_FOLDER

# Get all files in both folders
mri_files = [os.path.join(mri_folder, f) for f in os.listdir(mri_folder) 
            if f.endswith('.nii.gz')]

annotation_files = [os.path.join(annotation_folder, f) for f in os.listdir(annotation_folder) 
                if f.endswith('.nii.gz')]

# Match MRI files with corresponding annotation files
file_pairs = match_files(mri_files, annotation_files)


logger.info(f"Found {len(file_pairs)} matching pairs out of {len(mri_files)} MRI files and {len(annotation_files)} annotation files")

if __name__ == "__main__":

    mri_folder = MRI_FOLDER
    annotation_folder = ANNOTATION_FOLDER
    
    # Get all files in both folders
    mri_files = [os.path.join(mri_folder, f) for f in os.listdir(mri_folder) 
                if f.endswith('.nii.gz')]
    
    annotation_files = [os.path.join(annotation_folder, f) for f in os.listdir(annotation_folder) 
                       if f.endswith('.nii.gz')]
    
    # Match MRI files with corresponding annotation files
    file_pairs = match_files(mri_files, annotation_files)
    
    logger.info(f"Found {len(file_pairs)} matching pairs out of {len(mri_files)} MRI files and {len(annotation_files)} annotation files")

    all_man_review_cases = []
    all_man_review_node_pairs = []
    
    all_matted_cases = []
    all_matted_nodes = []
    all_matted_statuses = []

    # mri_path = "data/raw/images/1077-T2_FS_TRA+301.nii.gz"
    # annotation_path = "data/raw/labels/1077-T2_FS_TRA+301.nii.gz"

    for mri_path, annotation_path in tqdm(file_pairs, desc="Processing file pairs", unit="pair"):
        logger.info(f"............Starting analysis for {mri_path} and {annotation_path}")

        try:
    
            adjacency_graph, node_pairs_man_review, node_labels = run_pipeline_on_case(mri_path=mri_path, annotation_path=annotation_path, debug=DEBUG)

            int_node_pairs_man_review = [(int(a), int(b)) for a, b in node_pairs_man_review]
            all_man_review_cases.extend([mri_path] * len(int_node_pairs_man_review))
            all_man_review_node_pairs.extend(int_node_pairs_man_review)
             

            len_node_labels = len(node_labels)
            logger.debug(f"node labels is {node_labels}")

            logger.info(f"Ran pipeline on case, starting to find node groups")

            nodes_to_merge = find_node_groups(adjacency_graph)

            output_filename = f"{os.path.basename(mri_path)}"

            merged_annotation, min_matted_list, matted_list = merge_annotations(nodes_to_merge, annotation_path, len_node_labels)

            logger.debug(f"min matted list is {min_matted_list}")
            logger.debug(f"matted list is {matted_list}")

            mat_or_remov = [" "] * len_node_labels # matted or removed

            for i in range(len(matted_list)):
                if matted_list[i]:
                    mat_or_remov[i] = "removed"

            for i in range(len(min_matted_list)):
                if min_matted_list[i]:
                    mat_or_remov[i] = "matted"

            logger.debug(f"mat or remov is {mat_or_remov}")

            all_matted_cases.extend([mri_path] * len_node_labels)
            all_matted_nodes.extend(node_labels)
            all_matted_statuses.extend(mat_or_remov)

            output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, output_filename)
            sitk.WriteImage(merged_annotation, output_path)

            logger.info(f"Successfully processed {mri_path}")

        except Exception as e:
            logger.error(f"Error processing {mri_path}: {str(e)}")
            continue


    if all_man_review_cases:
        man_review_df = pd.DataFrame({
            "Case": all_man_review_cases,
            "Node pair": all_man_review_node_pairs
        })
        man_review_df.to_csv('all_man_review_df.csv', index=False)
    
    if all_matted_cases:
        matted_df = pd.DataFrame({
            "Case": all_matted_cases,
            "Node": all_matted_nodes,
            "Matted": all_matted_statuses
        })
        matted_df.to_csv('all_matted_df.csv', index=False)
    
    logger.info(f"Processing complete. Processed {len(file_pairs)} file pairs.")