from transformers import MaskFormerForInstanceSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, MaskFormerImageProcessor
import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageOps
from random import randint

class WasteMask2Former(nn.Module):
    """
    Custom Mask RCNN for Waste
    """
    def __init__(self,
                 num_classes:int,
                 checkpoint_path: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.processor = MaskFormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
        self.model = self.__get_model_instance_segmentation__(self.num_classes)

        # If model checkpoint
        if checkpoint_path:
            self.checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
    

    def forward(self,inputs,targets=None):
        pass

    def __get_model_instance_segmentation__(self, num_classes):
        # Load model with configuration
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-ade-semantic",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        # Ensure output hidden states are enabled
        model.config.output_hidden_states = True
        model.config.use_auxiliary_loss = True

        return model


    def preprocessing(self, images):
        
        images = [ImageOps.exif_transpose(img) for img in images]
        images = [np.array(img) for img in images]

        transforms = A.Compose([
            A.Resize(height=512, width=512),
            ToTensorV2()
        ])
        
        images = transforms(images=images)["images"]
        return images
    
        
    def evaluate(self, images:list, idx2class:dict, targets:list=None, preprocessing=True):
        raw_images = images.copy()
        if preprocessing:
            images = self.preprocessing(images=images)
        #print(f"images: {images}")
        target_sizes = [img.shape[-2:] for img in images]

        #print(f"id2class: {idx2class}")
        
        #inst2cls = {0: 0, 1:1}
        #for i, label in enumerate(idx2class.keys()):
        #    inst2cls[i] = label

        #print(f"inst2cls: {inst2cls}")

        print(self.processor)
        inputs = self.processor(
            images=[img.numpy() for img in images],
            #instance_id_to_semantic_id=inst2cls,
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
            do_normalize=False
        ).to("cpu")
        print(f"inputs: {inputs}")
        
        inputs["pixel_values"] = inputs["pixel_values"].float() / 255.0 

        self.model.to("cpu")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

            pred_maps = self.processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )
            print(f"pred_maps: {pred_maps}")

        processed_images = []
        for i, pred_map in enumerate(pred_maps):
            # Overlay original image with masks
            mask = pred_map['segmentation']
            segments_info = pred_map['segments_info']
            print(f"mask.unique: {np.unique(mask)}")
            print(f"segments_info: {segments_info}")
            #print(f"pred_maps.shape: {mask.shape}")
            
            n_detections = len(mask.unique())
            print(f"Nr of detections: {n_detections}")
            print(f"mask.unique(): {mask.unique()}")

            seg_map = self.draw_segmentation_map(mask, segments_info, idx2class)
            result = self.__image_overlay__(images[i], seg_map)


            # result.show()
            
            processed_images.append(result)
        
        processed_pred_maps = []
        for pred in pred_maps:
            processed_pred_maps.append({
                "segmentation": pred["segmentation"].tolist(),
                "segments_info": pred["segments_info"],
            })
        #print(processed_pred_maps)

        return processed_pred_maps, processed_images
    
    def draw_segmentation_map(self, labels, segments_info, idx2class):
        """
        :param labels: Label array from the model.Should be of shape 
            <height x width>. No channel information required.
        :param palette: List containing color information.
            e.g. [[0, 255, 0], [255, 255, 0]] 
        """
        # create Numpy arrays containing zeros
        # later to be used to fill them with respective red, green, and blue pixels
        red_map = np.zeros_like(labels).astype(np.uint8)
        green_map = np.zeros_like(labels).astype(np.uint8)
        blue_map = np.zeros_like(labels).astype(np.uint8)

        segments_dict = {data["id"]: data["label_id"] for data in segments_info}
        print(f"segments_dict: {segments_dict}")
        for label_num in labels.unique():

            print(f"label_num:{int(label_num)}")
            if int(label_num) in segments_dict.keys():
                segment = segments_dict[int(label_num)]
            else:
                segment = -1
            print(f"segment: {segment}")
            if segment != 0 and segment != -1: print(f"Detected class: {idx2class[segment]}")
            
            index = (labels == label_num).cpu().detach().numpy()
            random_color = [randint(0, 255) for _ in range(3)]
            if label_num == -1:
                red_map[index] = np.array(0)
                green_map[index] = np.array(0)
                blue_map[index] = np.array(0)
            #elif segment == 0:
            #    red_map[index] = np.array(0)
            #    green_map[index] = np.array(0) 
            #    blue_map[index] = np.array(0)
            else:
                red_map[index] = np.array(random_color[0])
                green_map[index] = np.array(random_color[1])
                blue_map[index] = np.array(random_color[2])
            
            
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        
        return segmentation_map
    
    def __image_overlay__(self, image, segmented_image):
        """
        :param image: Image in RGB format.
        :param segmented_image: Segmentation map in RGB format. 
        """
        image = image
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))
        mask = Image.fromarray(segmented_image)
        image = Image.blend(image, mask, 0.85)
        
        return image
    
    def __random_palette__(self, n):
        """
        Generate random colors for the segmentation map.
        :param n: Number of classes.
        """
        palette = np.random.randint(0, 256, (n, 3), dtype=np.uint8).tolist()
        return palette

