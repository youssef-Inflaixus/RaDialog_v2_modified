import torch
from .model import ImageModel  # adjust import if needed
from .types import ImageModelOutput  # adjust based on your structure

def test_custom_encoder():
    # Set your custom encoder type (the one you added in types.py and encoder.py)
    img_encoder_type = "unetpp_efficientnet"  # <- Use your registered type

    # Create the model with this encoder
    model = ImageModel(
        img_encoder_type=img_encoder_type,
        joint_feature_size=128,  # or whatever size you set in your MLP projector
        freeze_encoder=False
    )

    # Set model to eval
    model.eval()

    # Create dummy image input [B, C, H, W]
    dummy_input = torch.randn(1, 3, 224, 224)

    # Run forward pass
    with torch.no_grad():
        output: ImageModelOutput = model(dummy_input)

    # Print output shapes
    print("Global embedding shape:", output.img_embedding.shape)  # [B, D]
    print("Patch embedding shape:", output.patch_embeddings.shape)  # [B, D, H', W']
    print("Projected global embedding:", output.projected_global_embedding.shape)  # [B, joint_feature_size]
    print("Projected patch embedding:", output.projected_patch_embeddings.shape)  # [B, joint_feature_size, H', W']

if __name__ == "__main__":
    test_custom_encoder()

