# unetpp_efficientnet.py
from typing import Any, List, Tuple, Type, Union
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import os

TypeSkipConnections = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class UNETPPEfficientNet(nn.Module):
    """Wrapper class for UNet++ architecture with EfficientNet backbone.

    The forward function is updated to return the penultimate layer activations,
    which are required to obtain image patch embeddings.
    """

    def __init__(self, encoder_name: str, encoder_weights: str = 'imagenet', 
                 in_channels: int = 3, num_classes: int = 1, **kwargs: Any) -> None:
        super(UNETPPEfficientNet, self).__init__()
        
        # Initialize UNet++ with EfficientNet backbone
        self.unetpp = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None
        )
    

    def forward(self, x: torch.Tensor, return_intermediate_layers: bool = False) -> Union[torch.Tensor, TypeSkipConnections]:
        """Forward pass for UNet++ EfficientNet model.
        
        :param return_intermediate_layers: If True, return intermediate layers from the encoder.
        """
        # Call encoder directly, which returns a list of feature maps
        features = self.unetpp.encoder(x)  # List[Tensor]

        # Usually: [x0, x1, x2, x3, x4] = [H/2, H/4, H/8, H/16, H/32]
        if return_intermediate_layers:
            return tuple(features[:5])  # If more layers exist, slice as needed
        else:
            return features[-1]  # Last (deepest) feature map
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint into the model."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        # Use 'model_state' based on the checkpoint structure
        if 'model_state' in checkpoint:
            self.load_state_dict(checkpoint['model_state'])
        else:
            raise KeyError("Expected 'model_state' in checkpoint dictionary.")
        
        print(f"Loaded checkpoint from {checkpoint_path}")


def unetpp_efficientnet(pretrained: bool = True, checkpoint_path: str = "/home/youssef/bone_fracture_detection/experiments/RaDialog_v2/LLAVA/biovil_t/best_checkpoint.pth", progress: bool = True, **kwargs: Any) -> UNETPPEfficientNet:
    """UNet++ with EfficientNet-4 as backbone with optional checkpoint loading."""
    encoder_weights = 'imagenet' if pretrained else None
    model = UNETPPEfficientNet(encoder_name='efficientnet-b4', encoder_weights=encoder_weights, **kwargs)

    # If a checkpoint is provided, load the custom checkpoint
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)  # Load the checkpoint

    return model
