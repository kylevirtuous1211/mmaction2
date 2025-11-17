import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS

@MODELS.register_module()
class HierarchicalExerciseLoss(nn.Module):
    """
    Hierarchical loss for the 9 EXERCISE classes (labels 0-8).
    It ignores the 'idle' class.
    
    Coarse categories are:
    0: lunge (covers 0, 1, 2)
    1: push_up (covers 3, 4, 5)
    2: squat (covers 6, 7, 8)
    """
    def __init__(self, coarse_penalty_weight=1.5, **kwargs):
        super().__init__()
        self.coarse_penalty_weight = coarse_penalty_weight
        
        # This tensor maps your 9 fine labels [0...8] to 3 coarse labels
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.register_buffer(
            'coarse_map',
            torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long)
        )
        self.num_coarse_classes = 4
        # Store kwargs (like class_weight) for the fine-grained loss
        self.fine_loss_kwargs = kwargs

    def forward(self,
                pred_logits, # Logits should have 9 classes
                label,       # Labels should be 0-8
                **kwargs):

        # --- 1. Fine-Grained Loss (Standard Cross Entropy) ---
        # This is the normal loss on your 9 classes
        loss_fine = F.cross_entropy(pred_logits, label, **self.fine_loss_kwargs)

        # --- 2. Coarse-Grained Loss ---
        coarse_label = self.coarse_map[label]

        pred_probs_fine = F.softmax(pred_logits, dim=1)
        
        batch_size = pred_logits.size(0)
        pred_probs_coarse = torch.zeros(
            (batch_size, self.num_coarse_classes), 
            device=pred_logits.device
        )
        
        coarse_map_expanded = self.coarse_map.repeat(batch_size, 1) # [B, 9]
        
        pred_probs_coarse.scatter_add_(
            dim=1, 
            index=coarse_map_expanded, 
            src=pred_probs_fine
        )
        
        loss_coarse = F.nll_loss(
            torch.log(pred_probs_coarse.clamp(min=1e-9)), 
            coarse_label
            # Don't pass kwargs here, class_weight would be the wrong shape
        )

        # --- 3. Combine Losses ---
        total_loss = loss_fine + (self.coarse_penalty_weight * loss_coarse)

        return total_loss