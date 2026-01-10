"""Vision Transformer model for BHL page classification."""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class BHLPageClassifier(nn.Module):
    """ViT-based multi-label classifier for BHL pages."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224", num_labels: int = 7):
        """Initialize the classifier.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
        """
        super().__init__()
        self.num_labels = num_labels

        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(model_name)

        # Multi-label classification head
        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor = None
    ) -> dict:
        """Forward pass.

        Args:
            pixel_values: Image tensor [batch_size, channels, height, width]
            labels: Multi-label targets [batch_size, num_labels]

        Returns:
            Dictionary with logits, loss (if labels provided), and probabilities
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=pixel_values)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]

        # Classification
        logits = self.classifier(pooled_output)

        # Multi-label probabilities (sigmoid for each class independently)
        probs = torch.sigmoid(logits)

        result = {
            'logits': logits,
            'probs': probs
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss

        return result

    def freeze_backbone(self):
        """Freeze ViT backbone for feature extraction only."""
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ViT backbone for fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True

    def get_attention_maps(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization.

        Args:
            pixel_values: Image tensor

        Returns:
            Attention weights tensor
        """
        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=True
        )
        return outputs.attentions
