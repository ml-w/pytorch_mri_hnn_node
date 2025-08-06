from pathlib import Path

file_path = "data/raw/images/1058-T2_FS_TRA+301.nii.gz"
extracted = Path(file_path).name.replace('.nii.gz', '')
print(extracted)