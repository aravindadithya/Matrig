import torch
import wandb
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import matplotlib.cm as cm
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients
import numpy as np
import contextlib

class CNNLogger:

    def __init__(self, inputs, targets, max_weight_filters=100, config=None):
        
        self.max_weight_filters = max_weight_filters
        self.non_critical = os.getenv('NON_CRITICAL_LOGS', 'False').lower() in ('true', '1', 't')
        self.viz_inputs = inputs
        self.viz_targets = targets
        self.layer_info = {}
        self.config = config

    def _normalize(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-7)

    @contextlib.contextmanager
    def _optimize_gradient_computation(self, net, inputs):
        # Enable gradients for Captum
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # Optimization: Freeze model parameters to avoid computing gradients for weights
        original_requires_grad = {}
        for name, param in net.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False
        
        # Ensure inputs require grad so the graph is built even if weights are frozen
        original_input_req_grad = inputs.requires_grad
        inputs.requires_grad = True
        
        try:
            yield
        finally:
            # Restore requires_grad
            for name, param in net.named_parameters():
                param.requires_grad = original_requires_grad.get(name, True)
            inputs.requires_grad = original_input_req_grad
            torch.set_grad_enabled(prev_grad_state)

    def _create_blended_images(self, inputs, attrs):
        imgs = []
        inputs = inputs.detach().cpu()
        attrs = attrs.detach().cpu()
        
        for i in range(len(inputs)):
            img = self._normalize(inputs[i])
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            
            attr = self._normalize(attrs[i])
            attr_np = attr.squeeze().numpy()
            
            heatmap = cm.viridis(attr_np)[:, :, :3]
            blended = img_np * heatmap
            imgs.append(wandb.Image(blended))
        return imgs

    def _create_heatmap_images(self, attrs):
        imgs = []
        attrs = attrs.detach().cpu()
        for i in range(len(attrs)):
            attr = self._normalize(attrs[i])
            attr_np = attr.squeeze().numpy()
            heatmap = cm.viridis(attr_np)[:, :, :3]
            imgs.append(wandb.Image(heatmap))
        return imgs

    def _create_diverging_images(self, inputs, attrs):
        imgs = []
        inputs = inputs.detach().cpu()
        attrs = attrs.detach().cpu()
        
        for i in range(len(inputs)):
            img = self._normalize(inputs[i])
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            
            # Convert image to grayscale to make the colored heatmap pop
            img_gray = np.mean(img_np, axis=2, keepdims=True)
            img_gray = np.repeat(img_gray, 3, axis=2)
            
            attr = attrs[i].squeeze().numpy()
            
            # Normalize symmetrically around 0 for diverging colormap
            limit = max(abs(attr.min()), abs(attr.max())) + 1e-7
            # Map [-limit, limit] to [0, 1] where 0 maps to 0.5 (White)
            attr_norm = (attr / (2 * limit)) + 0.5
            
            # bwr colormap: Blue (neg), White (0), Red (pos)
            heatmap = cm.bwr(attr_norm)[:, :, :3]
            
            # Blend: 0.5 image + 0.5 heatmap
            blended = 0.5 * img_gray + 0.5 * heatmap
            imgs.append(wandb.Image(blended))
        return imgs

    def log_weights(self, layer, layer_name):
        """Computes Conv2d weight visualization."""
        w = layer.weight.data
        op, ip, q, s = w.shape
        w = w.reshape(op * ip, 1, q, s)
        if w.shape[0] > self.max_weight_filters:
            w = w[:self.max_weight_filters]
        grid_w = vutils.make_grid(w, nrow=ip, normalize=True, scale_each=False)
        return {f"Weight_Filters/{layer_name}": wandb.Image(grid_w, caption=f"Weights_Filter of {layer_name}")}

    def compute_featuremap_visuals(self, layer_name, activation):
        """Computes Conv2d activation visualization."""
        if activation is not None:
            avg_act = activation.mean(dim=1, keepdim=True)
            imgs = self._create_heatmap_images(avg_act)
            return {f"Output/{layer_name}": imgs}
        return {}

    def compute_eigen_featuremap_visuals(self, layer, layer_name, inputs):
        
        #Get Weights and Flatten: (Out, In, H, W) -> (K, C*H*W)
        w = layer.weight.data
        k, c, h, kw = w.shape
        w_flat = w.reshape(k, -1)

        if w_flat.shape[1] > 5000:
            return {}

        # Compute Gram Matrix and Eigen Decomposition
        gram = torch.matmul(w_flat.t(), w_flat)

        try:
            vals, vecs = torch.linalg.eigh(gram)
        except RuntimeError:
            return {}

        num_vecs = min(10, vecs.shape[1])
        eigen_filters = vecs[:, -num_vecs:].t().reshape(num_vecs, c, h, kw)

        with torch.no_grad():
            out = F.conv2d(inputs, eigen_filters, stride=layer.stride, padding=layer.padding)
            
        avg_out = out.mean(dim=1, keepdim=True)
        imgs = self._create_heatmap_images(avg_out)
        return {f"Eigen_Featuremap_Mean/{layer_name}": imgs}

    def compute_grad_cam_visuals(self, net, layer, layer_name):

        if not self.config:
            return {}

        visuals = {}
        inputs = self.viz_inputs
        
        with self._optimize_gradient_computation(net, inputs):
            layer_gc = LayerGradCam(net, layer)
            num_classes = self.config.get('num_classes', 10)
            print(f"Computing GradCAM for {layer_name}...")

            for class_idx in range(num_classes):
                # GradCAM for current class
                attr = layer_gc.attribute(inputs, target=class_idx, relu_attributions=True)
                attr = LayerAttribution.interpolate(attr, inputs.shape[2:], interpolate_mode='bilinear')

                imgs = self._create_blended_images(inputs, attr)
                visuals[f"GradCAM-{class_idx}"] = imgs
        
        return visuals

    def compute_ig_visuals(self, net, pred_targets=None):

        if not self.config or pred_targets is None:
            return {}

        inputs = self.viz_inputs

        with self._optimize_gradient_computation(net, inputs):
            print("Computing Integrated Gradients...")
            ig = IntegratedGradients(net)
            attr = ig.attribute(inputs, target=pred_targets, n_steps=50, internal_batch_size=inputs.shape[0])
            # Sum values across channels to get a single heatmap per image (preserving sign)
            attr = torch.sum(attr, dim=1, keepdim=True)

            imgs = self._create_diverging_images(inputs, attr)
            return {"IG(Predicted)": imgs}
        
        return {}
        
    def update_layer_info(self, layer_name, layer, input, output):

        self.layer_info[layer_name] = {
            "layer": layer,
            "input": input,
            "output": output
        }

    def get_visuals(self, net=None, pred_targets=None):
        global_logs, per_sample_logs = {}, {}
        last_layer_name = list(self.layer_info.keys())[-1] if self.layer_info else None
        
        for layer_name, info in self.layer_info.items():
            per_sample_logs.update(self.compute_featuremap_visuals(layer_name, info['output']))
            if self.non_critical:
                global_logs.update(self.log_weights(info['layer'], layer_name))
                per_sample_logs.update(self.compute_eigen_featuremap_visuals(info['layer'], layer_name, info['input']))

        if net and last_layer_name:
            per_sample_logs.update(self.compute_grad_cam_visuals(net, self.layer_info[last_layer_name]['layer'], last_layer_name))
        if net and pred_targets is not None:
            per_sample_logs.update(self.compute_ig_visuals(net, pred_targets=pred_targets))
        
        return global_logs, per_sample_logs