import torch


class ScaledLoss(torch.nn.Module):
    """
    Scaled loss function
    as in paper 'SLAW: Scaled Loss Approximate Weighting for Efficient Multi-Task Learning'
    """

    def __init__(
        self, weights, beta: float = 1.0, epsilon: float = 1e-5, device="cuda"
    ):
        super().__init__()
        self.beta = beta
        self.epsilon = torch.Tensor([epsilon], device=device)
        self.weights = weights
        self.a = torch.zeros_like(
            weights, device=device, dtype=torch.float, requires_grad=False
        )
        self.b = torch.zeros_like(
            weights, device=device, dtype=torch.float, requires_grad=False
        )

    def move_to_device(self, device: str):
        self.epsilon = self.epsilon.to(device)
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.weights = self.weights.to(device)

    def forward(self, losses: torch.Tensor):
        self.a = self.beta * self.a.detach() + (1 - self.beta) * losses
        self.b = self.beta * self.b.detach() + (1 - self.beta) * losses**2

        s = torch.sqrt(torch.max(self.a - self.b**2, self.epsilon))
        weights = losses.shape[0] * s.sum() / s * self.weights
        return weighted_loss(losses, weights)


def weighted_loss(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (losses * weights).sum() / weights.sum()
