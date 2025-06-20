import cv2
import numpy as np
import torch.nn as nn
import torch


class GradCam:
    """
    Computes Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps for a specified target layer in a model.

    This implementation is designed for Vision Transformers and assumes the model returns token sequences
    where the first token is the [CLS] token and the rest correspond to patches.

    Attributes:
        model (torch.nn.Module): The model for which Grad-CAM is to be computed.
        target (torch.nn.Module): The specific layer to register hooks on (usually an attention block).
        feature (torch.Tensor): Stores the feature maps from the forward hook.
        gradient (torch.Tensor): Stores the gradients from the backward hook.
        handlers (list): Stores hook handler references for cleanup.
    """

    def __init__(self, model: nn.Module, target: nn.Module):
        """
        Initializes the GradCam object and registers hooks on the target layer.

        Args:
            model (torch.nn.Module): The model to evaluate.
            target (torch.nn.Module): The layer to extract features and gradients from.
        """
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = target
        self._get_hook()

    def _get_features_hook(self, module, input, output):
        """
        Hook function to store the forward-pass features.

        Args:
            module (nn.Module): The module hooked.
            input (torch.Tensor): Input tensor.
            output (torch.Tensor): Output tensor from the forward pass.
        """
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        Hook function to store the backward-pass gradients.

        Args:
            module (nn.Module): The module hooked.
            input_grad (tuple): Gradients of the inputs.
            output_grad (tuple): Gradients of the outputs.
        """
        self.gradient = self.reshape_transform(output_grad[0])

    def _get_hook(self):
        """Registers the forward and backward hooks on the target layer."""
        self.handlers.append(self.target.register_forward_hook(self._get_features_hook))
        self.handlers.append(self.target.register_full_backward_hook(self._get_grads_hook))

    def reshape_transform(self, tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
        """
        Reshapes transformer output to 3D spatial format (C, H, W).

        Args:
            tensor (torch.Tensor): Input tensor of shape (B, N, C) where N includes CLS token.
            height (int): Height of the spatial patch grid.
            width (int): Width of the spatial patch grid.

        Returns:
            torch.Tensor: Reshaped tensor of shape (B, C, H, W).
        """
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2) # (B, C, H, W)
        return result

    def __call__(self, inputs: torch.Tensor) -> np.ndarray:
        """
        Computes the Grad-CAM heatmap for a given input tensor.

        Args:
            inputs (torch.Tensor): Input image tensor of shape (1, C, H, W).

        Returns:
            np.ndarray: Heatmap of shape (224, 224) normalized between 0 and 1.
        """
        self.model.zero_grad()
        output = self.model(inputs)

        index = np.argmax(output.detach().cpu().numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].detach().cpu().numpy() # (C, H, W)
        weight = np.mean(gradient, axis=(1, 2)) # Global average pooling on gradients
        feature = self.feature[0].detach().cpu().numpy() # (C, H, W)

        # Weighted sum of features by gradient weights
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)
        cam = np.maximum(cam, 0) # ReLU

        cam -= np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam /= cam_max
        else:
            cam[:] = 0

        cam = cv2.resize(cam, (224, 224)) # Resize to input image size
        return cam
