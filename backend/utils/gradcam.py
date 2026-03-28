"""
Grad-CAM Heatmap Generator
===========================
Highlights which regions of the ECG triggered the AI prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from utils.model import get_model, IMAGE_TRANSFORM


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        # Hook into EfficientNet's last conv layer
        target_layer = self.model.backbone.features[-1]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, image_array: np.ndarray, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        device = next(self.model.parameters()).device
        
        pil_img = Image.fromarray(image_array.astype(np.uint8))
        tensor = IMAGE_TRANSFORM(pil_img).unsqueeze(0).to(device)
        tensor.requires_grad = True
        
        # Forward pass
        self.model.eval()
        output = self.model(tensor)
        
        if target_class is None:
            target_class = output.argmax().item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]           # (C, H, W)
        activations = self.activations[0]       # (C, H, W)
        
        weights = gradients.mean(dim=(1, 2))    # Global average pooling
        cam = torch.zeros(activations.shape[1:], device=device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize and resize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam_resized = cv2.resize(cam, (224, 224))
        
        return cam_resized


def generate_heatmap_overlay(image_array: np.ndarray, predicted_class_idx: int) -> dict:
    """
    Generate heatmap and overlay on original ECG image.
    Returns base64-encoded images.
    """
    import base64
    
    model = get_model()
    grad_cam = GradCAM(model)
    
    # Generate CAM
    cam = grad_cam.generate(image_array, target_class=predicted_class_idx)
    
    # Create colored heatmap
    heatmap = np.uint8(255 * cam)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on original
    original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    original_resized = cv2.resize(original_bgr, (224, 224))
    overlay = cv2.addWeighted(original_resized, 0.55, heatmap_colored, 0.45, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    def to_base64(arr):
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
    
    return {
        "heatmap_base64": to_base64(heatmap_rgb),
        "overlay_base64": to_base64(overlay_rgb),
        "original_base64": to_base64(image_array)
    }
