from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import nibabel as nib
import numpy as np
import os

TASK_NAME = "Lesions"
NNUNET_DATASET_ID = "750"
FOLDER_NAME = f"Dataset{NNUNET_DATASET_ID}_{TASK_NAME}"
INPUT_IMAGES = "./nifti_resources"
INPUT_MASKS = "./download_masks"
MAPPING_FILE = f"{FOLDER_NAME}_mapping.txt"


def clean_path(p):
    return p.strip().strip("'").strip('"')


def merge_and_save_masks(mask_paths, mask_dst, reference_affine):
    merged_data = None

    for mask_path in mask_paths:
        nii = nib.load(mask_path)
        data = nii.get_fdata()

        if data.ndim == 4 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)

        if data.ndim != 3:
            raise ValueError(f"Invalid mask shape {data.shape} in {mask_path}")

        if not np.allclose(data, np.round(data)):
            raise ValueError(f"Non-integer values in mask {mask_path}")

        data = data.astype(np.uint8)

        if merged_data is None:
            merged_data = data.copy()
        else:
            if data.shape != merged_data.shape:
                raise ValueError(f"Shape mismatch in {mask_path}")

            merged_data = np.logical_or(merged_data, data)

    nib.save(
        nib.Nifti1Image(merged_data.astype(np.uint8), reference_affine),
        mask_dst
    )


def convert_to_nnunet_format(input_images, input_masks, output_base):
    output_folder = join(output_base, FOLDER_NAME)

    imagesTr = join(output_folder, "imagesTr")
    labelsTr = join(output_folder, "labelsTr")
    imagesTs = join(output_folder, "imagesTs")
    num_test_cases = 0

    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)

    ''' map masks to images '''
    mask_files = [clean_path(join(input_masks, f))
                  for f in os.listdir(input_masks) if f.endswith(".nii.gz")]

    mask_map = {}
    for mf in mask_files:
        base = os.path.basename(mf)
        key = base.split("_segmentation")[0]
        mask_map.setdefault(key, []).append(mf)

    image_files = [clean_path(join(input_images, f))
                   for f in os.listdir(input_images) if f.endswith(".nii.gz")]

    train_cases = []
    case_counter = 1

    with open(MAPPING_FILE, "w", encoding="utf-8") as mapf:
        mapf.write("case_id | image_path | mask_paths\n")

        for img_path in image_files:
            base = os.path.basename(img_path)
            key = base.replace(".nii.gz", "")

            case_id = f"case_{case_counter:06d}"

            if key in mask_map:
                img_dst = join(imagesTr, f"{case_id}_0000.nii.gz")
                shutil.copy(img_path, img_dst)

                img_nii = nib.load(img_path)
                img_affine = img_nii.affine

                mask_dst = join(labelsTr, f"{case_id}.nii.gz")
                merge_and_save_masks(mask_map[key], mask_dst, img_affine)

                train_cases.append(case_id)

                mapf.write(
                    f"{case_id} | {img_path} | {', '.join(mask_map[key])}\n"
                )
            else:
                img_dst = join(imagesTs, f"{case_id}_0000.nii.gz")
                shutil.copy(img_path, img_dst)

                num_test_cases += 1
                mapf.write(
                    f"{case_id} | {img_path} | NONE\n"
                )

            case_counter += 1

    print("Report ")
    print(f"Saved mapping to {MAPPING_FILE}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Test cases: {num_test_cases}")

    return train_cases


if __name__ == "__main__":
    train_cases = convert_to_nnunet_format(INPUT_IMAGES, INPUT_MASKS, nnUNet_raw)

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

