from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

TASK_NAME = "Lesions"
NNUNET_DATASET_ID = "750"
FOLDER_NAME = f"Dataset{NNUNET_DATASET_ID}_{TASK_NAME}"
INPUT_IMAGES = "./nifti_resources"
INPUT_MASKS = "./downloaded_masks"

def extract_case_id_from_image(filename: str) -> str:
    return filename.replace(".nii.gz", "")

def extract_case_id_from_mask(filename: str) -> str:
    return filename.split("_segmentation")[0]

def convert_to_nnunet_format(input_images, input_masks, output_base):
    output_folder = join(output_base, FOLDER_NAME)

    imagesTr_folder = join(output_folder, "imagesTr")
    labelsTr_folder = join(output_folder, "labelsTr")
    imagesTs_folder = join(output_folder, "imagesTs")

    maybe_mkdir_p(imagesTr_folder)
    maybe_mkdir_p(labelsTr_folder)
    maybe_mkdir_p(imagesTs_folder)
    
    ''' Mapping case IDs to mask filenames '''
    mask_files = [f for f in os.listdir(input_masks) if f.endswith(".nii.gz")]
    mask_map = {}
    for mf in mask_files:
        case_id = extract_case_id_from_mask(mf)
        mask_map[case_id] = mf
    
    print(f"Found {len(mask_map)} masks.")
    
    image_files = [f for f in os.listdir(input_images) if f.endswith(".nii.gz")]
    train_cases = []
    test_cases = []
    
    for img_file in image_files:
        case_id = extract_case_id_from_image(img_file)

        img_src = join(input_images, img_file)

        if case_id in mask_map:
            ''' if there is a corresponding mask, it's a training case '''
            train_cases.append(case_id)

            img_dst = join(imagesTr_folder, f"{case_id}_0000.nii.gz")
            mask_src = join(input_masks, mask_map[case_id])
            mask_dst = join(labelsTr_folder, f"{case_id}.nii.gz")

            shutil.copy(img_src, img_dst)
            shutil.copy(mask_src, mask_dst)
        else:
            ''' otherwise, it's a test case '''
            test_cases.append(case_id)

            img_dst = join(imagesTs_folder, f"{case_id}_0000.nii.gz")
            shutil.copy(img_src, img_dst)
    
    print("Report: ")
    print(f"Total images processed: {len(image_files)}")
    print(f"Training cases: {len(train_cases)}")
    print(f"Testing cases: {len(test_cases)}")
    
    return train_cases
            

if __name__ == "__main__":
    train_cases = convert_to_nnunet_format(INPUT_IMAGES, INPUT_MASKS, nnUNet_raw, TASK_NAME, NNUNET_DATASET_ID)
    
    generate_dataset_json(
        join(nnUNet_raw, FOLDER_NAME),
        channel_names={0: "CT"},
        labels={
            "background": 0,
            "lesion": 1
        },
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=TASK_NAME
    )