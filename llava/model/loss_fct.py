import torch
from torch.nn.modules.loss import _WeightedLoss
from typing import Optional


# main idea: penalize the model more on </CN> and </UN> tokens
class WeightedIgnoreCrossEntropyLoss(_WeightedLoss):

    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    ignore_index: int
    label_smoothing: float
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        cnun_penalty_weight: float = 2.0,
        special_tokens: list = [128258],
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.cnun_penalty_weight = cnun_penalty_weight
        self.special_tokens = special_tokens

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # Get a mask corresponding to the ignore index
        mask = target == self.ignore_index
        target_abs_with_ignore = target.abs().masked_fill(mask, self.ignore_index)

        # Compute the loss using the abs value of the target
        loss_unreduced = torch.nn.functional.cross_entropy(
            input,
            target_abs_with_ignore,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )

        # Mask the loss corresponding to the ignore index
        loss_unreduced = loss_unreduced.masked_fill(mask, 0)
        # Apply penalty for special tokens
        for special_token in self.special_tokens:
            loss_unreduced[target == special_token] *= self.cnun_penalty_weight

        # Reduce the loss
        if self.reduction == "mean":
            # Return the mean of the non-masked loss values
            return loss_unreduced.sum() / (mask.numel() - mask.sum())
        elif self.reduction == "sum":
            return loss_unreduced.sum()

        return loss_unreduced


# Main idea
# Loss = 
#  - CrossEntropyLoss for positive target values (normal cross-entropy loss for positive next token prediction)
#  - Negative log likelihood for negative target values (prevent the model from generating hallucinated tokens)
# class NegativeIgnoreCrossEntropyLoss(_WeightedLoss):

#     __constants__ = ["ignore_index", "reduction", "label_smoothing"]
#     ignore_index: int
#     label_smoothing: float

#     def __init__(
#         self,
#         weight: Optional[torch.Tensor] = None,
#         size_average=None,
#         ignore_index: int = -100,
#         reduce=None,
#         reduction: str = "mean",
#         label_smoothing: float = 0.0,
#         min_log_likelihood: float = -8.0,
#         apply_cnun_penalty: bool = False,
#         cnun_penalty_weight: float = 2.0,
#         special_token: int = 128258,
#     ) -> None:
#         super().__init__(weight, size_average, reduce, reduction)
#         # arguments for cross-entropy loss
#         self.ignore_index = ignore_index
#         self.label_smoothing = label_smoothing
#         # prevent the model from optimizing the log likelihood to -inf by clamping it to a minimum value
#         self.min_log_likelihood = min_log_likelihood
#         # penalty for CN and UN tokens
#         self.apply_cnun_penalty = apply_cnun_penalty
#         if self.apply_cnun_penalty:
#             self.positive_loss_fct = WeightedIgnoreCrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none', label_smoothing=label_smoothing, cnun_penalty_weight=cnun_penalty_weight, special_tokens=special_token)
#         else:
#             self.positive_loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none', label_smoothing=label_smoothing)

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

#         # Get a mask corresponding to the ignore index
#         ignore_mask = target == self.ignore_index
#         target_abs_with_ignore = target.abs().masked_fill(ignore_mask, self.ignore_index)

#         # Compute the loss using the abs value of the target
#         loss_unreduced = self.positive_loss_fct(input, target_abs_with_ignore)

#         # Mask the loss corresponding to the ignore index
#         loss_unreduced = loss_unreduced.masked_fill(ignore_mask, 0)

#         # Compute loss for negative target values
#         negative_mask = (target < 0) & (~ignore_mask)

#         # check probability of the target classes (negative log likelihood for class[target])
#         log_probs = torch.nn.functional.log_softmax(input, dim=1)

#         neg_log_likelihood_loss = torch.gather(log_probs, 1, torch.where(negative_mask, target.abs(), torch.zeros_like(target)).unsqueeze(1)).squeeze()
#         # Clamp the negative log likelihood loss where target values are negative
#         clamped_neg_losses = torch.clamp(neg_log_likelihood_loss, min=self.min_log_likelihood)
#         final_loss_unreduced = torch.where(negative_mask, clamped_neg_losses, loss_unreduced)

#         # Reduce the loss
#         if self.reduction == "mean":
#             # Return the mean of the non-masked loss values
#             return final_loss_unreduced.sum() / (target.numel() - ignore_mask.sum())
#         elif self.reduction == "sum":
#             return final_loss_unreduced.sum()

#         return final_loss_unreduced
# class Optimized
class NegativeIgnoreCrossEntropyLoss(_WeightedLoss):
    __constants__ = ["ignore_index", "reduction", "label_smoothing"]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        k: int = 3,
        apply_cnun_penalty: bool = False,
        cnun_penalty_weight: float = 2.0,
        special_token: int = 128258,
    ) -> None:
        """
        Args:
            weight: Manual rescaling weight for each class.
            ignore_index: Specifies a target value that is ignored.
            reduction: Specifies the reduction to apply: 'mean', 'sum', or 'none'.
            label_smoothing: Amount of label smoothing.
            k: Top-k threshold for negative samples.
            apply_cnun_penalty: If True, applies an extra penalty on positive tokens matching special_token.
            cnun_penalty_weight: Multiplicative penalty factor for the special token.
            special_token: The token id to penalize (when apply_cnun_penalty is True).
        """
        super().__init__(weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.k = k
        self.apply_cnun_penalty = apply_cnun_penalty
        self.cnun_penalty_weight = cnun_penalty_weight
        self.special_token = special_token

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Logits tensor of shape (N, C).
            target: Target tensor of shape (N,). Negative values indicate that we want to penalize
                    the model if the target is among the top-k predictions.
        Returns:
            The computed loss.
        """
        # Masks for ignored and valid positions.
        ignore_mask   = (target == self.ignore_index)
        valid_mask    = ~ignore_mask
        positive_mask = (target >= 0) & valid_mask
        negative_mask = (target < 0) & valid_mask

        # Use absolute value of target for indexing.
        target_abs = target.abs().masked_fill(ignore_mask, 0)

        # Compute log-softmax once.
        log_probs = torch.nn.functional.log_softmax(input, dim=1)  # shape: (N, C)

        # -------------------------
        # Positive Branch
        # -------------------------
        tmp_value = log_probs.gather(1, target_abs.unsqueeze(1)).squeeze(1)
        if self.label_smoothing == 0:
            pos_loss = -tmp_value
        else:
            nll_loss   = -tmp_value
            smooth_loss = -log_probs.mean(dim=1)
            pos_loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss

        if self.weight is not None:
            pos_loss = pos_loss * self.weight[target_abs]

        # Optionally apply extra penalty for special_token in positive samples.
        if self.apply_cnun_penalty:
            penalty_mask = positive_mask & (target == self.special_token)
            pos_loss[penalty_mask] *= self.cnun_penalty_weight

        # -------------------------
        # Negative Branch
        # -------------------------
        # For negative targets, we want to check if the target's log-probability is among the top-k.
        # If it is, we penalize it by the difference between its log-probability and the k-th highest log-prob.
        neg_loss = torch.zeros_like(target, dtype=log_probs.dtype, device=log_probs.device)
        if negative_mask.any():
            # Compute target log-probabilities.
            target_log_prob = tmp_value

            # For each sample, get the top-k log-probabilities.
            topk_vals, topk_indices = torch.topk(log_probs, k=self.k, dim=1)
            kth_vals = topk_vals[:, -1]  # The k-th highest log-prob for each sample.

            # Check if the target is among the top-k.
            # target_abs is (N,) -> (N, 1) compared against topk_indices (N, k)
            in_topk = (topk_indices == target_abs.unsqueeze(1)).any(dim=1)

            # For samples where the negative target is in top-k, compute the penalty loss.
            # Loss is the amount by which the target's log_prob exceeds the k-th highest log_prob.
            loss_vals = target_log_prob - kth_vals
            # Only keep positive differences.
            neg_loss = torch.where(in_topk, torch.nn.functional.relu(loss_vals), torch.zeros_like(loss_vals))
            # Zero out samples not in the negative mask.
            neg_loss = neg_loss * negative_mask.to(neg_loss.dtype)

        # -------------------------
        # Combine Losses
        # -------------------------
        # For positive samples use pos_loss; for negative samples use neg_loss.
        loss = torch.where(negative_mask, neg_loss, pos_loss)
        # Zero out ignored positions.
        loss = loss.masked_fill(ignore_mask, 0)

        # -------------------------
        # Reduction
        # -------------------------
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss