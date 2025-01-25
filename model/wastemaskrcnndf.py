#https://pytorch.org/vision/0.19/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html#torchvision.models.detection.maskrcnn_resnet50_fpn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn

    
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict

class MaskRCNNdf(nn.Module):
    def __init__(self, backbone=None, num_classes=91,
                    # transform parameters
                    min_size=800, max_size=1333,
                    # RPN parameters
                    rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                    rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                    rpn_nms_thresh=0.7,
                    rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                    rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                    # Box parameters
                    box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                    box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                    box_batch_size_per_image=512, box_positive_fraction=0.25,
                    bbox_reg_weights=None,
                    ):
        super().__init__()
        # First, set the backbone that we will use to extract the feature maps
        self.backbone = backbone

        out_channels = backbone.out_channels

        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        
        predictor = nn.Sequential(OrderedDict([
                ("conv5_mask", nn.ConvTranspose2d(mask_predictor_in_channels,mask_dim_reduced,kernel_size=(2,2),stride=2)),
                ("relu", nn.ReLU(inplace=True)),
                ("mask_fcn_logits", nn.Conv2d(mask_dim_reduced,num_classes,kernel_size=(1,1),padding=1))
            ]))
        
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,mask_dim_reduced,num_classes)

        out_channels = backbone.out_channels
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        # Declare the anchor generator of the Region Proposal Network (RPN)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        # Declare the Head of the RPN that will classify the proposals
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        rpn_pre_nms_top_n = {"training": rpn_pre_nms_top_n_train, "testing": rpn_pre_nms_top_n_test}
        rpn_post_nms_top_n = {"training": rpn_post_nms_top_n_train, "testing": rpn_post_nms_top_n_test}

        # Create the RPN combining the anchor generator and the head
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # ROI Align
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        # Simple Head made of 2 Fully Connected layers
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        representation_size = 1024

        # Predictor with 2 outputs (scores and Bounding Boxes deltas)
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        # Module that combines the last modules to compute the prediction of every box and match
        # targets and proposalt in training
        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        min_size=800
        max_size=1333
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        # Module that does the correct data augmentation
        self.transform = GeneralizedRCNNTransform(min_size, max_size, self.image_mean, self.image_std)

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return detections, losses
    







