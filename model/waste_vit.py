import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class WasteViT(ViTForImageClassification):

    def __init__(self, num_classes=None, id2label=None, label2id=None, checkpoint=None):

        if checkpoint:
            # Load model directly from the checkpoint directory
            model = ViTForImageClassification.from_pretrained(checkpoint)

            # Extract metadata
            num_classes = model.config.num_labels
            id2label = model.config.id2label
            label2id = model.config.label2id

        if num_classes is None or id2label is None or label2id is None:
            raise ValueError("num_classes, id2label, and label2id must be provided if no checkpoint is given.")

        # Initialize base ViT model with correct config
        super().__init__(config=ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        ).config)

        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

        # If checkpoint provided, load model weights
        if checkpoint:
            self.load_state_dict(model.state_dict(), strict=False)

    def forward(self, pixel_values, labels=None):
        outputs = super().forward(pixel_values=pixel_values, labels=labels)
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

