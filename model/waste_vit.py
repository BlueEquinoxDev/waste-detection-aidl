import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class WasteViT(ViTForImageClassification):

    def __init__(self, num_classes=None, id2label=None, label2id=None, checkpoint=None):
        if checkpoint:
            
            # Load the model's configuration
            config = ViTConfig.from_pretrained(checkpoint)
            
            # Initialize with the config object
            super().__init__(config=config)
        
            self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            
            # Load weights
            state_dict = ViTForImageClassification.from_pretrained(checkpoint).state_dict()
            self.load_state_dict(state_dict, strict=True)  # Use strict=True to catch mismatches
        else:
            # Initialize new model with provided parameters
            if num_classes is None or id2label is None or label2id is None:
                raise ValueError("num_classes, id2label, and label2id must be provided if no checkpoint is given.")
            
            config = ViTConfig.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )

            config.attention_probs_dropout_prob = 0.48699113095794067
            config.hidden_dropout_prob = 0.48699113095794067
            
            super().__init__(config=config)
            self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    def forward(self, pixel_values, labels=None):
        outputs = super().forward(pixel_values=pixel_values, labels=labels)
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

