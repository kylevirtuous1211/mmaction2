# `mmaction2` Submodule: Action Recognition

This project relies heavily on the **`mmaction2`** toolbox. We have forked the repository to include several key modifications for our FIAS framework:

* **Custom Configs** for ST-GCN++ models.
* **Automated Scripts** for training, testing, and analysis.
* A **Custom Hierarchical Loss** function to improve coarse-level accuracy.
* **Custom Data Augmentations** (`randomRot`, `randomScale`).

---

### 1. Download Checkpoints

To run inference or fine-tune, you must first download the pre-trained model weights.

1.  **Find a Model:** Visit the [MMAction2 Model Zoo](https://mmaction2.readthedocs.io/en/latest/model_zoo/skeleton.html).
    * *Example:* `stgcnpp_80e_ntu60_xsub_keypoint_3d_e0e41362.pth`
2.  **Create Directory:** Create a `checkpoints/` folder inside this `mmaction2/` directory.
3.  **Place Checkpoint:** Download the `.pth` file and place it inside that folder.

The structure should be:
```
fitness-project/
├── mmaction2/
│   ├── checkpoints/
│   │   └── stgcnpp_80e_ntu60_xsub_keypoint_3d_e0e41362.pth
│   └── ... (other mmaction2 files)
```

## Custom config for ST-GCN++
To reproduce our experiments, make sure to use our config files.
## TODOs for customize to your own pipeline
* Most importantly, modify `ann_file_test`, `ann_file_train` to train on the annotation file built on the preprocessing steps (See readme.md at `preprocess/`)
* update the `num_classes` of the classifier head for your action recognition tasks types
* update `class_weight` based on your training class distribution. This is for reweighting technique

### Inheritance
Other modalities will inherit from `Joint` modality, therefore, you only need to update joint modality and the same config will be applied to all modality pipeline
```
/home/cvlab123/FIAS-A-Framework-for-Explainable-Skeleton-based-Action-Recognition-to-Empower-LLM-Fitness-Coaches/mmaction2/configs/skeleton/custom_stgcnpp
```

# Automated training / testing Script
## Training Script `/mmaction2/train_classifiers.sh`
This script will automatically train the 4 modalities (Joint, Joint Motion, Bone, Bone Motion) of the ST-GCN++ model. 

## Testing Script `/mmaction2/test_classifiers.sh`
This script will automatically test the 4 modalities (Joint, Joint Motion, Bone, Bone Motion) of the ST-GCN++ model based on the **best checkpoints** in the work_dirs directory.

Update this to point to the checkpoint saved path
```
# ❗️ Please verify this path matches your project structure.
CONFIG_BASE_PATH="configs/skeleton/custom_stgcnpp"
WORK_DIRS_BASE="work_dirs"
```
The file also dumps the prediction result to a .pkl format at this specified directory. This is for plotting Confusion Matrix and post analysis of the test results. 
```
local output_pkl_path="test_results/friend_test_${work_dir_suffix}.pkl"
```

# Automated Confusion Matrix Plotting pipeline 
```
cd test_results/
```
After running `test_classifiers.sh`, .pkl file of the testing results will be saved here automatically. 

Run `plot_confusion_matrix.py` for plot generation. But before you run, update the .pkl file path.
```
results_file = '/home/cvlab123/mmaction2/test_results/friend_test_1101_reweighting_add_idle_2D_joint_motion.pkl'
```

## Custom Hierarchical Loss
### Why is Hierarchical Loss Needed
Since our AI LLM coach's response heavily rely on the accuracy of our action recognition's accuracy. During the first few action recognition experiments, we realized that the model would predict wrong coarse actions. Therefore, we implemented an extra loss to penalize the model more for predicting wrong coarse action labels.

The path for custom loss is located at
```
/mmaction2/fias_custom_loss
```

The custom loss has already been imported in the config file. In case you are writing your own. You could add this loss to other config file, by including it as follows in the config
```
imports = ('fias_custom_loss.custom_loss')
```
## Custom augmentations
We also added custom augmentations: `randomRot` and `randomScale` of skeleton data

The augmentation implementation is located in:
```
mmaction2/mmaction/datasets/transforms/pose_transforms.py
```

The augmentation has already been included in our custom config file.
```
imports = ('mmaction.datasets.transforms.pose_transforms')
```

