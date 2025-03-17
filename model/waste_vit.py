import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class WasteViT(ViTForImageClassification):
    def __init__(self, num_classes=None, id2label=None, label2id=None, checkpoint=None):
        # Create configuration
        config = ViTConfig.from_pretrained(
            'google/vit-base-patch16-224', 
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id
        )
        
        # Initialize the parent class with this config
        super().__init__(config)
        
        # Initialize with pretrained weights
        if not checkpoint:
            self.load_weights_from_pretrained()
        
        # Customize classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.48699113095794067),
            nn.Linear(self.config.hidden_size, num_classes)
        )
        
        # Load feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # Load weights if provided
        if checkpoint and checkpoint.endswith('.pth'):
            self.load_state_dict(torch.load(checkpoint, map_location='cpu'))
            print(f"Loaded weights from {checkpoint}")
    
    def load_weights_from_pretrained(self):
        """Load pretrained weights excluding the classifier"""
        pretrained_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        pretrained_dict = pretrained_model.state_dict()
        
        # Filter out classifier parameters
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'classifier' not in k}
        
        # Update model weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    
    def forward(self, pixel_values, labels=None):
        outputs = super().forward(pixel_values=pixel_values, labels=labels)
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )