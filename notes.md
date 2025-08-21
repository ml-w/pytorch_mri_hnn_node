C9G677n<

IMAGES:
\\137.189.141.206\Shared\Matthew_Projects\Handover_Ching\Enlarged_Node_Segmentation\Datasets\NPC_Pre_nodes_images\images_normalized

DICOMS:
\\137.189.141.206\Shared\NPC\NPC_New_case\2005-2025_not into T1rho studies



1. check datasheet for id and remarks
2. annotate and measure the enlarged nodes one by one
(if needed, can check the T1 DICOM)
3. check for imaging nature from remarks
4. save segmentation (ctrl+s)

ctrl+j to increase brightness

what is the criteria for short axis -retrophar 6; 11



30-58, 21


data path:
\\137.189.141.206\Shared\Matthew_Projects\p30-NodeReporting\00.DATA\medstudent_Segmentation


setup env
- see if can use pytorch2
close nodes -> merge to one, label that its been merged
use bbox, no need polygon
classification -> some preprocessing?
- extract each annotation as one piece of data?
- n4 bias field correction?
- intensity normalization?
- resample to isotropic voxels


2d resnet train (modify mri_node_validator/models/detector.py), joi see if 3d is needed or not

1.85 + ssh switch to pre-release version


py 3.7.12
torch 1.11.0
cuda 11.6



v 1013: yellow and blue separate? blue is a matted node from 7-8?
v 1095: red and green?
v 1084: quite a lot are matted nodes
v 1077: green and red separate? yellow and cyan separate?

find some examples where merging is needed

merge criteria
- how close? contact patch how large? empty volume between them?
	- gap distance e.g. <1mm
	- contact area volume after e.g. 0.5mm dilation e.g. >30mm
	- contact area after dilation avg intensity vs each nodes intensity

and quickly skim to see if two nodes have been incorrectly merged together, extract each node, manually label if it is a matted node or truly a single node, or are they two nodes and have been incorrectly merged


current progress: tested that the 3 criteria have been properly implemented, tested with negative edge egs and it works well, will test with positive egs and scale up from file to folder
ask: 
- all criteria overall, across slices
- matted on certain slides, same colour on other slides? e.g. 931 yellow


correct approach:
use majority vote, majority of the slides is mated -> all are mated; majority separated -> all separated
half half -> flag for manual review

start training -> let dr wong know, managed w notion

this weeks goal: at least started training







ultimate aim: train a model to tell if can further separate
feed two types of data: cannot separate (original data, can be a single node or a matted node); can separate (augmented data, merge nodes that kinda look like matted nodes)

next steps:
# TODO 1. use merge_nodes.ipynb with normal criteria to check if some current annotations should actually be merged
# TODO (find from hugging face / yolo some mri pretrained models)
# TODO 2. check if the merged ones detected by 1 should actually be merged
# TODO 3. use merge_nodes.ipynb with much looser criteria to create a set of augmented data of nodes that can be further separated
# TODO 4. preprocess the original data and augmented data by separating the mri image into 2d images of the same size, one node per image; can choose an image size to feed, use scale up and scale down
# TODO 5. train

rgb put 3 slides

**note that only the "merged

merge nodes according to 3 criteria slice by slice + majority vote 
- used normal criteria and checked w justin
- used loose criteria to augment data
extract 3x 2d slices of nodes, worked but since i did it slice by slice and stacked them, each slice do not have the same size -> black border ----> maybe use max size those 3 slices instead? 
scale up works fine, just need to deal with the black border problem just mentioned

realized that the merge nodes criteria to create augmented data was a bit too strict ----> trying out different values for the parameters 

also found some pretrained weights pretrained on mri images but they look kinda sus, maybe try no pretrained weights as baseline first then try loading those pretrained weights


ask about centroid issue - different each slice / huge risk of losing part of the node
ask about the merge nodes criteria as well - 13x to 6x - class imbalance

v 1099 min distance 57 check why fried
v change the 3 slides to that slide and nearby slides, size determined by max among the ones with masks
v handle min max
v merge nodes - produce more

v cli for the prgms

kick becuz the program given does not seem to have a training part, only inference; requires annotation which would require me to modify previous programs and verify them again -> rewrote

v check output

# TODO: update create_labels.py so that files copied from each folder will have a prefix, then need to run that to create merged and need to run create p id col again

# TODO: see why x y at bottom left in itksnap is 1 224 not 1 1