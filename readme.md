note that slices id start with 0, while ITK-snap start with 1
# TODO fix y axis wrong direction in one of the preprocessing programs (merge or extract)

merge_nodes.py merges nodes with 3 tunable criteria: distance, area of contact region after dilation, intensity of contact region after dilation
extract_nodes_2dx3.py gets 2dx3 images of nodes
pad_to_size.py pads to a given size with the image randomly placed (not necessarily at the centre)
extract_merged_from_aug.py extracts only the merged nodes from a folder of augmented nodes
create_labels.py program to combine real individual and augmented nodes into one folder and record labels in a csv
split.py