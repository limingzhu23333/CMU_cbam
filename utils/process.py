import SimpleITK as sitk
import glob
import os

p = list(range(0, 200))
for name in p:
    # Dicom save to nii
    
   
    file_path = os.path.join("F:/data", f"{name}*/")
    matching_folders = glob.glob(file_path)

   
    if not matching_folders:
        print(f"No matching folders found for {file_path}")
        continue

   
    folder_path = matching_folders[0]

  
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
    if not series_IDs:
        print(f"No DICOM series found in {folder_path}")
        continue

  
    nb_series = len(series_IDs)
    print(nb_series)
    

   
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path, series_IDs[0])

  
    series_reader = sitk.ImageSeriesReader()

   
    series_reader.SetFileNames(series_file_names)

 
    image3D = series_reader.Execute()

  
    print(image3D.GetSize())


    sitk.WriteImage(image3D, f'F:/data/nii/{name}_1.nii')



import os
from dcmrtstruct2nii import dcmrtstruct2nii


path = 'D:/dose/99CHENGUANGMING/1.2.840.114358.0382.2.20240106183549.86651182322_rtss.dcm'
input_folder = 'D:/dose/99_1/'
output_folder = 'D:/dose/99_mask'  


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


dcmrtstruct2nii(path, input_folder, output_folder)

import gzip

def decompress_nii_gz(nii_gz_path, output_path):
    with gzip.open(nii_gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())


nii_gz_path = 'D:/dose/99_mask/mask_Tumor.nii.gz'
output_path = 'D:/dose/99_mask/mask_Tumor.nii'
decompress_nii_gz(nii_gz_path, output_path)



