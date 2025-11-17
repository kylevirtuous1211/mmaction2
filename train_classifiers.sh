#!/bin/bash
# This script trains eight ST-GCN++ models with different data modalities (joint, joint_motion, bone, bone_motion).
set -e

# --- Common paths ---
# ❗️ Please verify this path matches your project structure.
CONFIG_PATH="configs/skeleton/custom_stgcnpp" 

# --- Training Commands ---

echo "🚀 Starting training for joint model..."
python tools/train.py \
    "${CONFIG_PATH}/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    --work-dir work_dirs/1101_reweighting_add_idle_2D_joint

echo "🚀 Starting training for joint-motion model..."
python tools/train.py \
    "${CONFIG_PATH}/stgcnpp_8xb16-joint-motion-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    --work-dir work_dirs/1101_reweighting_add_idle_2D_joint_motion

echo "🚀 Starting training for bone model..."
python tools/train.py \
    "${CONFIG_PATH}/stgcnpp_8xb16-bone-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    --work-dir work_dirs/1101_reweighting_add_idle_2D_bone

echo "🚀 Starting training for bone-motion model..."
python tools/train.py \
    "${CONFIG_PATH}/stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-2d.py" \
    --work-dir work_dirs/1101_reweighting_add_idle_2D_bone_motion

# echo "🚀 Starting training for 3D joint model..."
# python tools/train.py \
#     "${CONFIG_PATH}/stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     --work-dir work_dirs/1101_CL_reweighting_add_idle_3D_joint

# echo "🚀 Starting training for joint-motion model..."
# python tools/train.py \
#     "${CONFIG_PATH}/stgcnpp_8xb16-joint-motion-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     --work-dir work_dirs/1101_CL_reweighting_add_idle_3D_joint_motion

# echo "🚀 Starting training for bone model..."
# python tools/train.py \
#     "${CONFIG_PATH}/stgcnpp_8xb16-bone-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     --work-dir work_dirs/1101_CL_reweighting_add_idle_3D_bone

# echo "🚀 Starting training for bone-motion model..."
# python tools/train.py \
#     "${CONFIG_PATH}/stgcnpp_8xb16-bone-motion-u100-80e_OurDataset-xsub-keypoint-3d.py" \
#     --work-dir work_dirs/1101_CL_reweighting_add_idle_3D_bone_motion


echo "✅ all training jobs completed successfully!"


# CUDA_VISIBLE_DEVICES=0 nohup bash train_classifiers.sh > 916_all.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup bash train_classifiers.sh > 916_all_lr=0.001_epoch=30.log 2>&1 &
#  tail -f 916_all_lr\=0.005_epoch\=30.log 