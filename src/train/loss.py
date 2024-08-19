import torch
from torch import nn
from torchvision.models import vgg16

def get_device():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, mse_weight=1.0, perceptual_weight=0.05):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.perceptual_weight = perceptual_weight
        self.mse_weight = mse_weight
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # Ensure input is three-channel by repeating grayscale image across three channels
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        mse_loss = self.mse_loss(pred, target)
        perceptual_loss = self.mse_loss(self.feature_extractor(pred_rgb), self.feature_extractor(target_rgb))
        return self.mse_weight * mse_loss + self.perceptual_weight * perceptual_loss

# Alternate loss
def perceptual_loss():
    # Initialize a feature extractor for Perceptual Loss
    feature_extractor = vgg16(pretrained=True).features[:16].eval().to(get_device())
    return PerceptualLoss(feature_extractor)