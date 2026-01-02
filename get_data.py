from datamint import Api 
import os
import dicom2nifti
import nibabel as nib
from medimgkit.readers import read_array_normalized
import numpy as np

PROJECT_NAME = "Renata study"
OUTPUT_RESOURCES = "./download_resources"
OUTPUT_MASK = "./download_masks"
OUTPUT_NIFTI = "./nifti_resources"

def get_resources_and_masks():
    ''' Connect to DataMint API and get project by name '''
    api = Api()

    print("Connected to DataMint API.")

    project = api.projects.get_by_name(PROJECT_NAME)

    if project is None:
        raise ValueError(f"Project '{PROJECT_NAME}' not found.")
    else :
        print(f"Project '{PROJECT_NAME}' found")
        
    resources = api.projects.get_project_resources(project)
    resources = resources[3:5]
    if resources:
        print(f"Found {len(resources)} resources in project '{PROJECT_NAME}'")
        
    os.makedirs(OUTPUT_RESOURCES, exist_ok=True)
    os.makedirs(OUTPUT_MASK, exist_ok=True)

    for resource in resources:
        base_name = os.path.splitext(resource.filename)[0]

        resource_bytes = resource.fetch_file_data(
            use_cache=True,
            auto_convert=False
        )

        im = read_array_normalized(resource_bytes)   # (N, C, H, W)
        im_nii = im[:, 0]                            # (Z, H, W)

        img_out = os.path.join(OUTPUT_RESOURCES, f"{base_name}.nii.gz")
        nib.save(nib.Nifti1Image(im_nii.astype(np.float32), np.eye(4)), img_out)

        print(f"Saved image NIfTI '{img_out}'")
        
        annotations = api.annotations.get_list(
        resource=resource,
        load_ai_segmentations=True
        )
        
        segmentation_annotation = [ann for ann in annotations if ann.type == 'segmentation']
        print(f"Found {len(segmentation_annotation)} segmentations for resource '{base_name}'")
        
        for count, seg in enumerate(segmentation_annotation):
            masks_bytes = seg.fetch_file_data(
                use_cache=True,
                auto_convert=False
            )

            masks = read_array_normalized(masks_bytes)  # (N, 1, H, W)
            masks = masks[::-1]                          
            mask_nii = masks[:, 0]

            seg_out = os.path.join(
                OUTPUT_MASK,
                f"{base_name}_segmentation_{count}.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(mask_nii.astype(np.uint8), np.eye(4)),
                seg_out
            )

            print(f"Saved segmentation '{seg_out}'")

    print("All resources and masks have been downloaded.")


def convert_dicom_to_nifti(dicom_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    subfolders = [f.path for f in os.scandir(dicom_folder) if f.is_dir()]
    
    for folder in subfolders:
        base_name = os.path.basename(folder)
        output_file = os.path.join(OUTPUT_NIFTI, f"{base_name}.nii.gz")
        
        try:
            dicom2nifti.dicom_series_to_nifti(folder, output_file, reorient_nifti=True)
            print(f"Successfully converted folder {base_name} to NIfTI")
        except Exception as e:
            print(f"Failed to convert {base_name}: {e}")
            
''' Call this function if previous conversion failed due to slice increment issues (SLICE_INCREMENT_INCONSISTENT error was raised) '''
def convert_dicom_with_previous_error(list_dicom_folder):
    import dicom2nifti.settings as settings
    settings.disable_validate_slice_increment()
    
    for folder in list_dicom_folder:
        base_name = os.path.basename(folder)
        output_file = os.path.join(OUTPUT_NIFTI, f"{base_name}.nii.gz")
        
        try:
            dicom2nifti.dicom_series_to_nifti(folder, output_file, reorient_nifti=True)
            print(f"Successfully converted folder {base_name} to NIfTI")
        except Exception as e:
            print(f"Failed to convert {base_name}: {e}")
    
    
if __name__ == "__main__":
    get_resources_and_masks()
    #convert_dicom_to_nifti(OUTPUT_RESOURCES, OUTPUT_NIFTI)
    
    #files_with_error = ["download_resources/D:\LESÕES ÓSSEAS OK\P266\20220608", "download_resources/D:\LESÕES ÓSSEAS OK\P249\20220531", 
    #                    "download_resources/D:\LESÕES ÓSSEAS OK\P246\20221116", "D:\LESÕES ÓSSEAS OK\P245\20221026", 
    #                    "D:\LESÕES ÓSSEAS OK\P157\20240828"]
    #convert_dicom_with_previous_error(files_with_error)
