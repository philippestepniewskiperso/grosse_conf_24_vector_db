import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation


class ClothSegmenter:
    def __init__(self):
        self.extractor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b0_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b0_clothes")

    def extract_mask_from_image(self, image):
        inputs = self.extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        if (pred_seg == 4).float().sum() > (pred_seg == 7).float().sum():
            specific_mask = (pred_seg == 4).float()
        else:
            specific_mask = (pred_seg == 7).float()
        mask_image = Image.fromarray((specific_mask.cpu().numpy() * 255).astype('uint8'))
        masked_image = Image.composite(image, Image.new('RGB', image.size, color=(211, 211, 211)), mask_image)
        return masked_image
