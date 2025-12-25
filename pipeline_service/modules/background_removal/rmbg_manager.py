from __future__ import annotations

import time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop

from config import Settings
from logger_config import logger


class BackgroundRemovalService:
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService.
        """
        self.settings = settings

        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding
        
        # Kamui Enhancement: Initialize enhanced settings
        self.mask_threshold = max(
            self.settings.mask_threshold_min,
            min(self.settings.mask_threshold_max, self.settings.mask_threshold)
        )
        self.use_smart_padding = self.settings.use_smart_padding
        self.use_adaptive_threshold = self.settings.use_adaptive_threshold
        self.enable_antialiasing = self.settings.enable_antialiasing
        self.enable_quality_metrics = self.settings.enable_quality_metrics

        # Set device
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"

        # Set model
        self.model: AutoModelForImageSegmentation | None = None

        # Set transform
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
            ]
        )

        # Set normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    async def startup(self) -> None:
        """
        Startup the BackgroundRemovalService.
        """
        logger.info(f"Loading {self.settings.background_removal_model_id} model...")

        # Load model
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.background_removal_model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            logger.success(f"{self.settings.background_removal_model_id} model loaded.")
        except Exception as e:
            logger.error(f"Error loading {self.settings.background_removal_model_id} model: {e}")
            raise RuntimeError(f"Error loading {self.settings.background_removal_model_id} model: {e}")

    async def shutdown(self) -> None:
        """
        Shutdown the BackgroundRemovalService.
        """
        self.model = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """
        Ensure the BackgroundRemovalService is ready.
        """
        if self.model is None:
            raise RuntimeError(f"{self.settings.background_removal_model_id} model not initialized.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from the image.
        """
        try:
            t1 = time.time()
            # Check if the image has alpha channel
            has_alpha = False
            
            if image.mode == "RGBA":
                # Get alpha channel
                alpha = np.array(image)[:, :, 3]
                if not np.all(alpha==255):
                    has_alpha=True
            
            if has_alpha:
                # If the image has alpha channel, return the image
                output = image
                image_without_background = output  # Fix: define variable in this path too
                
            else:
                # PIL.Image (H, W, C) C=3
                rgb_image = image.convert('RGB')
                
                # Tensor (H, W, C) -> (C, H',W')
                rgb_tensor = self.transforms(rgb_image).to(self.device)
                output = self._remove_background(rgb_tensor)

                image_without_background = to_pil_image(output[:3])

            removal_time = time.time() - t1
            logger.success(f"Background remove - Time: {removal_time:.2f}s - OutputSize: {image_without_background.size} - InputSize: {image.size}")

            return image_without_background
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return image 

    def _remove_background(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove the background from the image.
        """
        # Normalize tensor value for background removal model, reshape for model batch processing (C=3, H, W) -> (1, C=3, H, W)
        input_tensor = self.normalize(image_tensor).unsqueeze(0)
                
        with torch.no_grad():
            # Get mask from model (1, 1, H, W)
            preds = self.model(input_tensor)[-1].sigmoid()
            # Kamui Enhancement: Keep continuous mask values (no quantization for better edges)
            mask = preds[0].squeeze()
            
            # Apply quantization only if configured (0 = no quantization for best quality)
            if self.settings.mask_quantization_bits > 0:
                quant_levels = 2 ** self.settings.mask_quantization_bits - 1
                mask = mask.mul_(quant_levels).int().div(quant_levels).float()
        
        # Kamui Enhancement: Use adaptive or configured threshold
        threshold = self._calculate_threshold(mask) if self.use_adaptive_threshold else self.mask_threshold
        
        # Get bounding box indices
        bbox_indices = torch.argwhere(mask > threshold)
        
        # Kamui Enhancement: Validate object size
        if len(bbox_indices) == 0:
            crop_args = dict(top=0, left=0, height=mask.shape[1], width=mask.shape[0])
        elif not self._validate_object_size(bbox_indices, mask.shape):
            # Fallback to full image if object detection failed
            crop_args = dict(top=0, left=0, height=mask.shape[1], width=mask.shape[0])
        else:
            # Kamui Enhancement: Smart padding based on object shape
            if self.use_smart_padding:
                crop_args = self._calculate_smart_padding(mask, bbox_indices)
            else:
                # Original padding logic
                h_min, h_max = torch.aminmax(bbox_indices[:, 1])
                w_min, w_max = torch.aminmax(bbox_indices[:, 0])
                width, height = w_max - w_min, h_max - h_min
                center = (h_max + h_min) / 2, (w_max + w_min) / 2
                size = max(width, height)
                padded_size_factor = 1 + self.padding_percentage
                size = int(size * padded_size_factor)
                top = int(center[1] - size // 2)
                left = int(center[0] - size // 2)
                bottom = int(center[1] + size // 2)
                right = int(center[0] + size // 2)

                if self.limit_padding:
                    top = max(0, top)
                    left = max(0, left)
                    bottom = min(mask.shape[1], bottom)
                    right = min(mask.shape[0], right)

                crop_args = dict(
                    top=top,
                    left=left,
                    height=bottom - top,
                    width=right - left
                )
            
            # Kamui Enhancement: Log centering quality if enabled
            if self.settings.log_centering_quality and 'center_of_mass' in crop_args:
                centering = self._validate_centering_quality(
                    crop_args, mask.shape, crop_args.pop('center_of_mass')
                )
                if centering['quality_score'] < 0.8:
                    logger.warning(
                        f"Object centering quality: {centering['quality_score']:.2f}, "
                        f"offset: {centering['offset_pixels']}"
                    )

        mask = mask.unsqueeze(0)
        # Concat mask with image and blacken the background: (C=3, H, W) | (1, H, W) -> (C=4, H, W)
        tensor_rgba = torch.cat([image_tensor*mask, mask], dim=-3)
        
        # Kamui Enhancement: Enable antialiasing for smooth edges
        output = resized_crop(
            tensor_rgba, 
            top=crop_args['top'],
            left=crop_args['left'],
            height=crop_args['height'],
            width=crop_args['width'],
            size=self.output_size, 
            antialias=self.enable_antialiasing
        )
        
        # Kamui Enhancement: Calculate quality metrics if enabled
        if self.enable_quality_metrics:
            metrics = self._calculate_quality_metrics(mask.squeeze(), output)
            logger.info(f"BG removal quality: confidence={metrics['mask_confidence_mean']:.3f}, "
                       f"coverage={metrics['alpha_coverage']:.2%}, "
                       f"edge_sharpness={metrics['edge_sharpness']:.3f}")
        
        return output
    
    # ============ Kamui Enhancement Methods ============
    
    def _calculate_threshold(self, mask: torch.Tensor) -> float:
        """
        Calculate adaptive threshold using Otsu's method.
        
        Args:
            mask: Binary mask tensor (H, W)
            
        Returns:
            Optimal threshold value
        """
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            
            # Convert to numpy and flatten
            mask_np = mask.cpu().numpy().flatten()
            
            # Use Otsu's method (maximize inter-class variance)
            hist, bins = np.histogram(mask_np, bins=256, range=(0, 1))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            
            mean1 = np.cumsum(hist * bin_centers) / weight1
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
            
            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            
            idx = np.argmax(variance)
            otsu_threshold = bin_centers[idx]
            
            # Clamp to configured range
            threshold = max(
                self.settings.mask_threshold_min,
                min(self.settings.mask_threshold_max, otsu_threshold)
            )
            
            logger.debug(f"Adaptive threshold: {threshold:.3f} (Otsu: {otsu_threshold:.3f})")
            return threshold
            
        except Exception as e:
            logger.warning(f"Adaptive threshold failed, using default: {e}")
            return self.mask_threshold
    
    def _validate_object_size(self, bbox_indices: torch.Tensor, mask_shape: tuple) -> bool:
        """
        Validate that detected object is within reasonable size bounds.
        
        Args:
            bbox_indices: Object bounding box indices
            mask_shape: Shape of the mask (H, W)
            
        Returns:
            True if object size is valid
        """
        if len(bbox_indices) == 0:
            return False
            
        object_pixels = len(bbox_indices)
        total_pixels = mask_shape[0] * mask_shape[1]
        coverage = object_pixels / total_pixels
        
        min_coverage = self.settings.min_object_coverage
        max_coverage = self.settings.max_object_coverage
        
        if coverage < min_coverage:
            logger.warning(f"Object too small: {coverage:.2%} < {min_coverage:.2%}")
            return False
        elif coverage > max_coverage:
            logger.warning(f"Object too large: {coverage:.2%} > {max_coverage:.2%}")
            return False
            
        return True
    
    def _calculate_smart_padding(self, mask: torch.Tensor, bbox_indices: torch.Tensor) -> dict:
        """
        Calculate adaptive padding based on object shape and position.
        
        Args:
            mask: Binary mask tensor (H, W)
            bbox_indices: Object bounding box indices
            
        Returns:
            Dictionary with crop parameters
        """
        h_min, h_max = torch.aminmax(bbox_indices[:, 1])
        w_min, w_max = torch.aminmax(bbox_indices[:, 0])
        width, height = w_max - w_min, h_max - h_min
        
        # Calculate center of mass (more accurate than bbox center)
        mask_cpu = mask.cpu()
        y_indices, x_indices = torch.where(mask_cpu > self.mask_threshold)
        if len(y_indices) > 0:
            center_y = y_indices.float().mean()
            center_x = x_indices.float().mean()
            center_of_mass = (center_y.item(), center_x.item())
        else:
            center_of_mass = ((h_max + h_min) / 2).item(), ((w_max + w_min) / 2).item()
        
        # Detect object shape (aspect ratio)
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Adaptive padding factor based on shape
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            # Elongated object needs more padding
            padding_factor = self.padding_percentage * self.settings.adaptive_padding_factor
        else:
            # Roughly square object uses normal padding
            padding_factor = self.padding_percentage
        
        size = max(width, height)
        padded_size_factor = 1 + padding_factor
        size = int(size * padded_size_factor)
        
        # Use center of mass for better centering
        top = int(center_of_mass[0] - size // 2)
        left = int(center_of_mass[1] - size // 2)
        bottom = int(center_of_mass[0] + size // 2)
        right = int(center_of_mass[1] + size // 2)
        
        if self.limit_padding:
            top = max(0, top)
            left = max(0, left)
            bottom = min(mask.shape[0], bottom)
            right = min(mask.shape[1], right)
        
        crop_args = dict(
            top=top,
            left=left,
            height=bottom - top,
            width=right - left,
            center_of_mass=center_of_mass  # Pass for quality validation
        )
        
        logger.debug(
            f"Smart padding: aspect={aspect_ratio:.2f}, "
            f"padding={padding_factor:.2%}, CoM=({center_of_mass[1]:.0f},{center_of_mass[0]:.0f})"
        )
        
        return crop_args
    
    def _validate_centering_quality(self, crop_args: dict, mask_shape: tuple, 
                                   center_of_mass: tuple) -> dict:
        """
        Validate that object is well-centered in the crop.
        
        Args:
            crop_args: Crop parameters
            mask_shape: Shape of the mask (H, W)
            center_of_mass: Object center of mass (y, x)
            
        Returns:
            Dictionary with centering quality metrics
        """
        crop_center_y = crop_args['top'] + crop_args['height'] / 2
        crop_center_x = crop_args['left'] + crop_args['width'] / 2
        
        offset_y = abs(crop_center_y - center_of_mass[0])
        offset_x = abs(crop_center_x - center_of_mass[1])
        offset_pixels = (offset_y ** 2 + offset_x ** 2) ** 0.5
        
        # Quality score (1.0 = perfect centering, 0.0 = worst)
        max_offset = min(crop_args['height'], crop_args['width']) / 2
        quality_score = 1.0 - min(1.0, offset_pixels / max_offset)
        
        return {
            'quality_score': quality_score,
            'offset_pixels': offset_pixels,
            'offset_y': offset_y,
            'offset_x': offset_x
        }
    
    def _calculate_quality_metrics(self, mask: torch.Tensor, output: torch.Tensor) -> dict:
        """
        Calculate quality metrics for background removal result.
        
        Args:
            mask: Binary mask tensor (H, W)
            output: Output RGBA tensor (4, H, W)
            
        Returns:
            Dictionary with quality metrics
        """
        # Mask confidence (mean and std of mask values)
        mask_values = mask[mask > self.mask_threshold]
        mask_confidence_mean = mask_values.mean().item() if len(mask_values) > 0 else 0.0
        mask_confidence_std = mask_values.std().item() if len(mask_values) > 0 else 0.0
        
        # Alpha channel coverage
        alpha_channel = output[3]
        alpha_coverage = (alpha_channel > 0.5).float().mean().item()
        
        # Edge sharpness (gradient magnitude on alpha channel)
        dy = alpha_channel[1:] - alpha_channel[:-1]
        dx = alpha_channel[:, 1:] - alpha_channel[:, :-1]
        edge_sharpness = (dy.abs().mean() + dx.abs().mean()).item() / 2
        
        return {
            'mask_confidence_mean': mask_confidence_mean,
            'mask_confidence_std': mask_confidence_std,
            'alpha_coverage': alpha_coverage,
            'edge_sharpness': edge_sharpness
        }


