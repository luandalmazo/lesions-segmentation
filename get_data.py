from datamint import Api 
import os
import dicom2nifti
import nibabel as nib

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
    if resources:
        print(f"Found {len(resources)} resources in project '{PROJECT_NAME}'")
        
    os.makedirs(OUTPUT_RESOURCES, exist_ok=True)
    os.makedirs(OUTPUT_MASK, exist_ok=True)

    for resource in resources:
        base_name = resource.filename.replace('\\', '/').split('/')[-1]
        base_name = os.path.splitext(base_name)[0]
        
        resource_dir = os.path.join(OUTPUT_RESOURCES, base_name)
        os.makedirs(resource_dir, exist_ok=True)
        
        resource_output_path = os.path.join(resource_dir, f"{base_name}.dcm")
        
        im = resource.fetch_file_data(use_cache=True, save_path=resource_output_path)
        print(f"Downloaded resource '{base_name}' to '{resource_output_path}'")
        
        annotations = api.annotations.get_list(
        resource=resource,
        load_ai_segmentations=True
        )
        
        segmentation_annotation = [ann for ann in annotations if ann.type == 'segmentation']
        
        print(f"Found {len(segmentation_annotation)} segmentations for resource '{base_name}'")
        
        if segmentation_annotation:
            count = 0
            for seg in segmentation_annotation:
                segmentation_output_name = f"{base_name}_segmentation_{count}.nii.gz"
                seg_data = seg.fetch_file_data(use_cache=True, save_path=os.path.join(OUTPUT_MASK, segmentation_output_name))
                print(type(seg_data ))
                count += 1
                print(f"Downloaded segmentation annotation '{segmentation_output_name}' to '{OUTPUT_MASK}'")

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
    
if __name__ == "__main__":
    get_resources_and_masks()
    convert_dicom_to_nifti(OUTPUT_RESOURCES, OUTPUT_NIFTI)