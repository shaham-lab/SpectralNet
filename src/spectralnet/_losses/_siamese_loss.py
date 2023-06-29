import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the contrastive loss between the two outputs of the siamese network.

        Parameters
        ----------
        output1 : torch.Tensor
            The first output of the siamese network.
        output2 : torch.Tensor
            The second output of the siamese network.
        label : torch.Tensor
            The label indicating whether the two outputs are similar (1) or not (0).

        Returns
        -------
        torch.Tensor
            The computed contrastive loss value.

        Notes
        -----
        This function takes the two outputs `output1` and `output2` of the siamese network,
        along with the corresponding `label` indicating whether the outputs are similar (1) or not (0).
        The contrastive loss is computed based on the Euclidean distance between the outputs and the label,
        and the computed loss value is returned.
        """

        euclidean = nn.functional.pairwise_distance(output1, output2)
        positive_distance = torch.pow(euclidean, 2)
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2)
        loss = torch.mean(
            (label * positive_distance) + ((1 - label) * negative_distance)
        )
        return loss
