#!/bin/bash
# This script tests four ST-GCN++ models by automatically finding the
# 'best_acc_top1' checkpoint in their respective work directories.
# It intelligently selects the MOST RECENTLY modified checkpoint if multiple exist.
# MODIFIED: It now saves the path of the found checkpoint to a .txt file.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# ❗️ Please verify this path matches your project structure.
CONFIG_BASE_PATH="configs/skeleton/custom_stgcnpp"
WORK_DIRS_BASE="work_dirs"

# --- Helper Function to Run a Single Test ---
# This function finds the best checkpoint, saves its path, and runs the test.
run_test() {
    # Arguments passed to the function
    local model_config_name=$1
    local work_dir_suffix=$2
    local model_description=$3
    local feature_type=$4 # New argument for naming the output .txt file

    echo "--- Preparing to test ${model_description} model ---"

    # Construct the full paths
    local config_file="${CONFIG_BASE_PATH}/${model_config_name}"
    local work_dir="${WORK_DIRS_BASE}/${work_dir_suffix}"

    # Check if the work directory exists
    if [ ! -d "$work_dir" ]; then
        echo "❌ ERROR: Work directory not found at ${work_dir}"
        echo "Please ensure the model has been trained first."
        return 1
    fi

    # Find the MOST RECENT checkpoint file starting with 'best_acc_top1'.
    # This command finds all matching files, sorts them by modification time (newest first),
    # and selects the top one. This is robust if you have multiple checkpoint files.
    local checkpoint_file
    checkpoint_file=$(find "${work_dir}" -name "best_acc_top1*.pth" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
    # checkpoint_file="/home/cvlab123/mmaction2/work_dirs/custom_stgcnpp_optuna/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d/trial_34/best_acc_top1_epoch_6.pth"

    # Check if a checkpoint file was found
    if [ -z "$checkpoint_file" ]; then
        echo "❌ ERROR: No 'best_acc_top1*.pth' checkpoint found in ${work_dir}"
        echo "Please check if training completed successfully and a best model was saved."
        return 1
    fi

    # --- MODIFICATION: Save the checkpoint path to a file ---
    # The 'realpath' command converts the relative path to an absolute path for clarity.
    local absolute_checkpoint_path
    absolute_checkpoint_path=$(realpath "${checkpoint_file}")
    local output_file="../best_checkpoint_${feature_type}.txt"
    
    echo "📝 Saving checkpoint path to ${output_file}"
    echo "${absolute_checkpoint_path}" > "${output_file}"
    # --- End of Modification ---

    echo "✅ Found newest checkpoint: ${checkpoint_file}"
    echo "🚀 Running test for ${model_description}..."

    # --- MODIFICATION: Added --dump argument to save results ---
    # This creates a structured output path for the .pkl file.
    local output_pkl_path="test_results/test_${work_dir_suffix}.pkl"

    # Execute the test command and pipe the output to grep.
    # This filters the verbose logging to show only the final result line.
    # The '|| true' prevents the script from exiting if grep finds no match. General command structure
    python tools/test.py "${config_file}" "${checkpoint_file}" --dump "${output_pkl_path}" | grep "Epoch(test)" || true

    echo "--- Test for ${model_description} finished ---"
    echo "" # Add a blank line for better readability
}


# --- Main Execution ---
# The script will now call the run_test function for each of your four models.
# A fourth argument (e.g., "joint", "joint_motion") has been added for the output filename.

# 915_rtmpose_front
# 916_rtmpose_all
# 916_rtmpose_back


run_test \
    "stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    "1101_CL_reweighting_add_idle_2D_joint" \
    "Joint" \
    "joint"

run_test \
    "stgcnpp_8xb16-joint-motion-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    "1101_CL_reweighting_add_idle_2D_joint_motion" \
    "Joint Motion" \
    "joint_motion"

run_test \
    "stgcnpp_8xb16-bone-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    "1101_CL_reweighting_add_idle_2D_bone" \
    "Bone" \
    "bone"

run_test \
    "stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    "1101_CL_reweighting_add_idle_2D_bone_motion" \
    "Bone Motion" \
    "bone_motion"

# run_test \
#     "stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     "916_rtmpose_all_3D_joint" \
#     "Joint" \
#     "joint_3D"

# run_test \
#     "stgcnpp_8xb16-joint-motion-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     "916_rtmpose_all_3D_joint_motion" \
#     "Joint Motion" \
#     "joint_motion_3D"

# run_test \
#     "stgcnpp_8xb16-bone-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     "916_rtmpose_all_3D_bone" \
#     "Bone" \
#     "bone_3D"

# run_test \
#     "stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     "916_rtmpose_all_3D_bone_motion" \
#     "Bone Motion" \
#     "bone_motion_3D"

# finetune 2d both joint
# /home/cvlab123/mmaction2/work_dirs/custom_stgcnpp_optuna/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d/trial_34/best_acc_top1_epoch_6.pth

echo "✅ All testing jobs completed successfully!"