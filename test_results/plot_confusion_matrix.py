import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path
import os 

# --- Configuration ---
# 1. UPDATE THIS: The path to your .pkl file from the --dump command
MODALITY = '2D Joint Motion Mixed View'
results_file = '/home/cvlab123/mmaction2/test_results/friend_test_1101_reweighting_add_idle_2D_joint_motion.pkl'
# results_file = 'test_results/825_rtmpose_train_front_test_back_3D_bone_motion_bone_motion_3D.pkl'
# results_file = 'test_results/825_rtmpose_train_front_test_back_3D_bone_motion_bone_motion_3D.pkl'
# results_file = 'test_results/825_rtmpose_train_front_test_back_3D_bone_motion_bone_motion_3D.pkl'

CLASS_NAMES = {"lunge_correct": 0, "lunge_knee_pass_toe": 1, "lunge_too_high": 2, "push_up_arched_back": 3, "push_up_correct": 4, "push_up_elbow": 5, "squat_correct": 6, "squat_feet_too_close": 7, "squat_knees_inward": 8}
# CLASS_NAMES = {
#         "idle": 0,
#         "lunge_correct": 1,
#         "lunge_knee_pass_toe": 2,
#         "lunge_too_high": 3,
#         "push_up_arched_back": 4,
#         "push_up_correct": 5,
#         "push_up_elbow": 6,
#         "squat_correct": 7,
#         "squat_feet_too_close": 8,
#         "squat_knees_inward": 9
#     }

try:
    with open(results_file, 'rb') as f:
        # This loads the list of dictionaries
        predictions = pickle.load(f)
except FileNotFoundError:
    print(f"Error: The file '{results_file}' was not found.")
    print("Please make sure the path is correct.")
    exit()

print(f"Successfully loaded {len(predictions)} predictions from the file.")

# Extract the ground truth and predicted labels from the list of dictionaries
# The .item() method converts a single-element tensor (e.g., tensor([2])) to a standard Python number (e.g., 2)
y_true = [p['gt_label'].item() for p in predictions]
y_pred = [p['pred_label'].item() for p in predictions]

accuracy_score = accuracy_score(y_true, y_pred, normalize=True)
# Use scikit-learn to compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nComputed Confusion Matrix from the dumped file:")
print(cm)

label_names = sorted(CLASS_NAMES, key=CLASS_NAMES.get)

# --- Plotting the Matrix ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,          # Show the numbers in the cells
    fmt='d',             # Format numbers as integers
    cmap='Blues',        # Color scheme
    xticklabels=label_names,
    yticklabels=label_names
)

plt.title(f'Best single view model ({MODALITY}), Accuracy: {accuracy_score:.4f}', fontsize=16)
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()

# Save the plot to a file
os.makedirs('confusion_matrices', exist_ok=True) 

# Creates a name like 'confusion_matrices/816_rtmpose_both_2D_joint_cm.png'
output_image_path = f"confusion_matrices/{Path(results_file).stem}_cm.png"
plt.savefig(output_image_path)

print(f"\nSuccessfully saved the confusion matrix plot to: {output_image_path}")

cm_sum = cm.sum(axis=1)[:, np.newaxis] + 1e-7
cm_normalized = cm.astype('float') / cm_sum

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2f',          # <-- IMPORTANT: Format as 2-decimal-place float
    cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names,
    vmin=0.0,           # Set the color scale minimum to 0
    vmax=1.0            # Set the color scale maximum to 1
)

plt.title(f'Normalized Confusion Matrix (Recall)\n({MODALITY}), Accuracy: {accuracy_score:.4f}', fontsize=16)
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()

# Creates a new name like '..._cm_normalized.png'
output_image_path_norm = f"confusion_matrices/{Path(results_file).stem}_cm_normalized.png"
plt.savefig(output_image_path_norm)

print(f"Successfully saved the normalized matrix to: {output_image_path_norm}")