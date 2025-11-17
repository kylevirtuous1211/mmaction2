import pickle
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

# --- Configuration ---
BASELINE_PKL = '915_rtmpose_front_2D_joint.pkl'
BEST_2D_PKL_SINGLE = '915_rtmpose_front_2D_bone_motion.pkl'
BEST_2D_PKL_MIXED = '916_rtmpose_all_2D_bone_motion.pkl'
BEST_3D = '916_rtmpose_all_3D_joint_motion.pkl'

OUTPUT_CHART_FILENAME = 'f1_plot/f1_score_comparison.png'

CLASS_NAMES = [
    'lunge_correct', 'lunge_knee_pass_toe', 'lunge_too_high',
    'push_up_arched_back', 'push_up_correct', 'push_up_elbow',
    'squat_correct', 'squat_feet_too_close', 'squat_knees_inward'
]

def calculate_f1_from_dump(pkl_path, num_classes):
    """Loads prediction results from a .pkl file and calculates per-class F1 scores."""
    if not os.path.exists(pkl_path):
        print(f"⚠️ WARNING: File not found at {pkl_path}. Skipping.")
        return None

    with open(pkl_path, 'rb') as f:
        predictions = pickle.load(f)

    if not isinstance(predictions, list) or not all('gt_label' in p for p in predictions):
        print(f"❌ ERROR: The file {pkl_path} is not a prediction dump.")
        return None

    try:
        y_true = [p['gt_label'].item() for p in predictions]
        y_pred = [p['pred_label'].item() for p in predictions]
        print(y_pred)
    except Exception as e:
        print(f"❌ ERROR reading labels from {pkl_path}: {e}")
        return None

    return f1_score(y_true, y_pred, labels=range(num_classes), average=None)

def plot_f1_comparison(baseline_scores, best_2d_scores, best_3d_scores, single_view_scores, class_names):
    """Generates and saves a bar chart comparing F1 scores."""
    if any(s is None for s in [baseline_scores, best_2d_scores, best_3d_scores, single_view_scores]):
        print("❌ ERROR: Cannot generate plot due to missing data.")
        return

    num_classes = len(class_names)
    for arr in [baseline_scores, best_2d_scores, best_3d_scores, single_view_scores]:
        if len(arr) != num_classes:
            print(f"❌ ERROR: Score length mismatch. Expected {num_classes}, got {len(arr)}")
            return

    x = np.arange(num_classes)
    width = 0.2

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    rects1 = ax.bar(x - 1.5*width, baseline_scores, width, label='Baseline (2D joint, front)', color='#4c72b0')
    rects2 = ax.bar(x - 0.5*width, single_view_scores, width, label='Best Single-View (2D bone motion, front)', color='#55a868')
    rects3 = ax.bar(x + 0.5*width, best_3d_scores, width, label='Best 3D (3D joint motion, both)', color='#c44e52')
    rects4 = ax.bar(x + 1.5*width, best_2d_scores, width, label='Best 2D (2D bone motion, both)', color='#8172b3')

    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Exercise Class', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison (Baseline vs. 2D vs. 3D vs. Single-View)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=0, loc='upper center', bbox_to_anchor=(0, 1.12), ncol=1)

    for rects in [rects1, rects2, rects3, rects4]:
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    fig.tight_layout()
    plt.savefig(OUTPUT_CHART_FILENAME, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to {OUTPUT_CHART_FILENAME}")

def main():
    num_classes = len(CLASS_NAMES)
    baseline_f1 = calculate_f1_from_dump(BASELINE_PKL, num_classes)
    best_2d_f1 = calculate_f1_from_dump(BEST_2D_PKL_MIXED, num_classes)
    best_3d_f1 = calculate_f1_from_dump(BEST_3D, num_classes)
    best_single_view_f1 = calculate_f1_from_dump(BEST_2D_PKL_SINGLE, num_classes)

    plot_f1_comparison(baseline_f1, best_2d_f1, best_3d_f1, best_single_view_f1, CLASS_NAMES)

if __name__ == '__main__':
    main()
