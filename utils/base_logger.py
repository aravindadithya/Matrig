import torch
import wandb
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.amp import autocast
import io
import os
import math
import hickle as hkl
from utils.cnn_logger import CNNLogger
from utils.agop_fc import verify_NFA
from utils.linear_rfa import LinearRFA


class BaseLogger:

    def __init__(self, config):
        
        self.config = config
        self.optimizer = config.get('optimizer')
        self.lfn = config.get('lfn')
        self.net = config.get('net')
        self.scheduler = config.get('scheduler')
        self.task_type = config.get('task_type', 'classification')
        self.train_loader = config.get('train_loader')

        # Store initial network state for AGOP computation
        self.init_net = self._get_initial_net()

        self._initialize_wandb(config)

        self.start_epoch = 1
        self.best_val_acc = 0
        self.best_val_loss = float("inf")
        self.best_state_dict = None
        self.rotate_inputs = config.get('rotate_inputs', True)
        self.max_images = config.get('max_images', 32)

        self.inputs, self.targets = self.get_viz_inputs(config.get('val_loader'))

        if wandb.run.resumed:
            self._resume_run()
        else:
            self._log_initial_artifacts(self.inputs)

    def _get_initial_net(self):
        """Create a copy of the initial network state for AGOP computation."""
        import copy
        return copy.deepcopy(self.net)

    def _initialize_wandb(self, config):

        key = os.getenv('WANDB_API_KEY')
        if key:
            wandb.login(key=key)

        # Filter out non-serializable objects for WandB config
        exclude_keys = ['net', 'train_loader', 'val_loader', 'test_loader', 'optimizer', 'lfn', 'scheduler']
        wandb_config = {k: v for k, v in config.items() if k not in exclude_keys}

        wandb.init(project=config['project'], name=config['run_name'], resume="allow", id=config['run_id'], config=wandb_config, entity=config['entity'])

        # Define 'epoch' as the step metric for all epoch-level logs
        wandb.define_metric("epoch")
        if self.task_type == 'regression':
            metrics_to_sync = [
                "train/loss", "val/loss", "learning_rate", "fixed_val_images"
            ]
        else:
            metrics_to_sync = [
                "train/accuracy", "train/loss", "val/accuracy", "val/loss",
                "learning_rate", "Validation Confusion Matrix", "Validation Predictions",
                "Test Confusion Matrix",
                "Test Predictions", "fixed_val_images"
            ]
        wandb.define_metric("adjacent_balance/*", step_metric="epoch")
        wandb.define_metric("matrix_product/*", step_metric="epoch")
        wandb.define_metric("agop/*", step_metric="epoch")
        for metric in metrics_to_sync:
            wandb.define_metric(metric, step_metric="epoch")

    def get_viz_inputs(self, val_loader):

        try:
            if self.task_type == 'regression':
                collected_inputs, collected_targets = [], []
                for batch in val_loader:
                    batch_inputs, batch_targets = batch[:2]
                    for i in range(len(batch_inputs)):
                        if len(collected_inputs) < self.max_images:
                            collected_inputs.append(batch_inputs[i])
                            collected_targets.append(batch_targets[i])
                    if len(collected_inputs) >= self.max_images:
                        break

                if not collected_inputs:
                    print("Viz setup failed: Could not collect any samples from val_loader.")
                    return None, None

                inputs = torch.stack(collected_inputs).cuda(non_blocking=True)
                targets = torch.stack(collected_targets).cuda(non_blocking=True)
                return inputs, targets

            # Get num_classes from config, with a default.
            num_classes = self.config.get('num_classes')
            images_per_class = max(1, self.max_images // num_classes)
            target_num_images = images_per_class * num_classes

            class_counts = {i: 0 for i in range(num_classes)}
            collected_inputs, collected_targets = [], []

            # Iterate through the loader to find a balanced set of images
            for batch in val_loader:
                batch_inputs, batch_targets = batch[:2]
                for i in range(len(batch_inputs)):
                    label = batch_targets[i].item()
                    if label in class_counts and class_counts[label] < images_per_class:
                        collected_inputs.append(batch_inputs[i])
                        collected_targets.append(batch_targets[i])
                        class_counts[label] += 1

                if sum(class_counts.values()) >= target_num_images:
                    break

            if not collected_inputs:
                print("Viz setup failed: Could not collect any images from val_loader.")
                return None, None

            print(f"Collected {len(collected_inputs)} images for visualization ({images_per_class} from each of {num_classes} classes).")
            inputs = torch.stack(collected_inputs)
            targets = torch.stack(collected_targets)

            # Sort inputs and targets by class label
            indices = torch.argsort(targets)
            inputs = inputs[indices]
            targets = targets[indices]

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if inputs.dim() == 4 and self.rotate_inputs:
                # Rotate inputs: 0, 90, 180, 270
                rotated_inputs = []
                for rot in range(4):
                    rotated_inputs.append(torch.rot90(inputs, k=rot, dims=[2, 3]))       
                # Stack: (B, 4, C, H, W)
                inputs = torch.stack(rotated_inputs, dim=1)
                inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
                targets = torch.repeat_interleave(targets, 4, dim=0)
            return inputs, targets

        except Exception as e:
            print(f"Viz setup failed with an unexpected error: {e}")
            return None, None

    def _resume_run(self):
        print("Resuming from previous run...")
        self.best_val_acc = wandb.run.summary.get("best_val_accuracy", 0)
        self.best_val_loss = wandb.run.summary.get("best_val_loss", float("inf"))
        try:
            # Load checkpoint to resume training state
            artifact = wandb.use_artifact(f"checkpoint-{self.config['run_id']}:latest")
            self.start_epoch = artifact.metadata.get("epoch", 0) + 1
            path = artifact.get_entry('last_model.pth').download()
            checkpoint = torch.load(path, weights_only=True)
            self.net.load_state_dict(checkpoint['state_dict'])

            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            try:
                # Load the state dict from the model with the best validation accuracy
                best_model_artifact = wandb.use_artifact(f"model-{self.config['run_id']}:latest")
                best_model_path = best_model_artifact.get_entry('best_model.pth').download()
                best_model_checkpoint = torch.load(best_model_path, weights_only=True)
                self.best_state_dict = best_model_checkpoint['state_dict']
            except Exception:
                pass

            print(f"Best Val Acc so far in the training run: {self.best_val_acc}")
        except Exception as e:
            print(f"Failed to resume from checkpoint artifact: {e}")

    def _log_initial_artifacts(self, inputs):
        # Log initial weights for new runs
        artifact = wandb.Artifact(f"init-weights-{self.config['run_id']}", type='model', metadata={"epoch": 0})
        with artifact.new_file('init_model.pth', mode='wb') as f:
            torch.save({'state_dict': self.net.state_dict()}, f)
        wandb.log_artifact(artifact)
        
        if inputs is not None:
            try:
                self.net.eval()
                dummy_input = inputs[0].unsqueeze(0).cuda()
                artifact = wandb.Artifact(f"onnx-{self.config['run_id']}", type='model')
                onnx_buffer = io.BytesIO()
                torch.onnx.export(self.net, dummy_input, onnx_buffer, input_names=['input'], output_names=['output'])
                with artifact.new_file('model.onnx', mode='wb') as f:
                    f.write(onnx_buffer.getvalue())
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: ONNX export failed: {e}")
            finally:
                self.net.train()

    def log_confusion_matrix(self, y_true, preds, epoch, class_names, log_key):
        wandb.log({
            log_key: wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=preds,
                class_names=class_names),
            "epoch": epoch
        })

    def log_predictions_table(self, net, log_key, outputs_precomputed=None, extra_visuals=None):
        net.eval()
        inputs, targets = self.inputs, self.targets
        if inputs is None:
            return {}

        num_classes = self.config.get('num_classes', 10)
        columns = ["Image", "Ground Truth", "Prediction", "Confidence"] + [f"Score_{i}" for i in range(num_classes)]
        
        extra_keys = []
        if extra_visuals:
            extra_keys = list(extra_visuals.keys())
            columns.extend(extra_keys)

        table = wandb.Table(columns=columns)
        
        with torch.no_grad():
            if outputs_precomputed is not None:
                outputs = outputs_precomputed
            else:
                with autocast(device_type='cuda'):
                    outputs = net(inputs)

            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            if len(targets.size()) > 1:
                _, targets = torch.max(targets, -1)

            inputs_cpu = inputs.cpu()
            targets_cpu = targets.cpu()
            preds_cpu = preds.cpu()
            confidences_cpu = confidences.cpu()
            probs_cpu = probs.cpu()

            for j in range(len(inputs_cpu)):
                img = inputs_cpu[j]
                if img.dim() == 1:
                    # Dynamic reshape instead of hardcoded 28x28
                    side = int(math.sqrt(img.numel()))
                    if side * side == img.numel():
                        img = img.view(1, side, side)
                
                img_viz = vutils.make_grid(img, normalize=True)
                row = [wandb.Image(img_viz), str(targets_cpu[j].item()), str(preds_cpu[j].item()), confidences_cpu[j].item()]
                row.extend(probs_cpu[j].tolist())
                
                if extra_visuals:
                    for key in extra_keys:
                        if j < len(extra_visuals[key]):
                            row.append(extra_visuals[key][j])
                        else:
                            row.append(None)

                table.add_data(*row)

        if table.data:
            return {log_key: table}
        return {}

    def log_visuals(self, net, epoch):
        if self.inputs is None:
            return

        print("Generating Visuals...")
        net.eval()

        layer_handlers = {}
        # Set up a handler for Conv2d layers. This is fine.
        layer_handlers[torch.nn.Conv2d] = CNNLogger(self.inputs, self.targets, config=self.config, max_weight_filters=20)

        hooks = []
        def get_activation(name, layer, handler):
            def hook(model, input, output):
                inp = input[0] if isinstance(input, tuple) else input
                handler.update_layer_info(name, layer, inp.detach(), output.detach())
            return hook

        # This loop correctly filters for layers we have handlers for.
        for name, layer in net.named_modules():
            if type(layer) in layer_handlers:
                handler = layer_handlers[type(layer)]
                hooks.append(layer.register_forward_hook(get_activation(name, layer, handler)))

        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = net(self.inputs)

        for h in hooks:
            h.remove()

        all_logs = {}
        #print("Generating Layer Visuals (GradCAM, IG, FeatureMaps)...")
        pred_targets = torch.argmax(outputs, dim=1)
        per_sample_visuals = {}

        for handler in layer_handlers.values():
            # Only get visuals if the handler was actually used (i.e., it hooked some layers).
            if handler.layer_info:
                global_logs, sample_logs = handler.get_visuals(net=net, pred_targets=pred_targets)
                all_logs.update(global_logs)
                per_sample_visuals.update(sample_logs)

        # print("Logging Prediction Table...")
        # all_logs.update(self.log_predictions_table(net, "Validation Predictions", outputs_precomputed=outputs, extra_visuals=per_sample_visuals))

        if all_logs:
            all_logs["epoch"] = epoch
            wandb.log(all_logs)
        print("Visuals Logging Completed.")

    def log_agop(self, epoch):
        """Log AGOP metrics for each linear layer."""
        if self.train_loader is None or self.init_net is None:
            return

        # Get the sequential module in a network-agnostic way
        linear_layers = [
            m for m in self.net.features.modules()
            if isinstance(m, (torch.nn.Linear, LinearRFA))
        ]
        
        logs = {"epoch": epoch}
        
        for layer_idx in range(len(linear_layers)):
            try:
                agop_metrics = verify_NFA(
                    net=self.net,
                    init_net=self.init_net,
                    trainloader=self.train_loader,
                    layer_idx=layer_idx,
                    max_batch=10,
                    classes=784,
                    chunk_idx=8
                )
                # Log AGOP metrics for this layer
                for metric_name, value in agop_metrics.items():
                    logs[f"agop/layer_{layer_idx}/{metric_name}"] = value
                    
            except Exception as e:
                print(f"Failed to compute AGOP for layer {layer_idx}: {e}")
                continue

        if len(logs) > 1:  # More than just epoch
            wandb.log(logs)

    def count_sparsity(self, epoch):
        """Count and log sparsity metrics for the network."""
        eps = 1e-6

        total_params = 0
        total_non_zero = 0

        for module in self.net.modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                weight = module.weight.detach()
                layer_total = weight.numel()
                layer_non_zero = (weight.abs() > eps).sum().item()

                total_params += layer_total
                total_non_zero += layer_non_zero

        if total_params > 0:
            total_zero = total_params - total_non_zero
            wandb.log({
                "epoch": epoch,
                "sparsity/non_zero_count/total": total_non_zero,
                "sparsity/zero_count/total": total_zero,
            })

    def _load_target_matrix(self, device):
        """Load target matrix from config."""
        matrix_path = self.config.get("target_matrix_path")
        if not matrix_path:
            return None

        if not os.path.exists(matrix_path):
            print(f"Target matrix path not found: {matrix_path}")
            return None

        matrix_np = hkl.load(matrix_path)
        target_matrix = torch.as_tensor(matrix_np, dtype=torch.float32, device=device)
        return target_matrix

    def _compute_product_matrix(self, linear_weights):
        """Compute product matrix for layers [W1, W2, ..., WL]."""
        # For layers [W1, W2, ..., WL], the effective matrix is WL ... W2 W1.
        product = linear_weights[0]
        for w in linear_weights[1:]:
            product = w @ product
        return product

    def log_matrix_diagnostics(self, epoch):
        """Log matrix diagnostics for the network."""
        linear_weights = [ m.weight for m in self.net.modules() if 
            isinstance(m, (torch.nn.Linear, LinearRFA))
        ]

        if not linear_weights:
            return
    
        target_matrix = self._load_target_matrix(device='cuda')
        logs = {"epoch": epoch}

        with torch.no_grad():
            product_matrix = self._compute_product_matrix([w.detach() for w in linear_weights])
            if target_matrix is not None:
                if product_matrix.shape == target_matrix.shape:
                    logs["matrix_product/||product - target||_F"] = torch.linalg.norm(
                        product_matrix - target_matrix, ord='fro'
                    ).item()
                else:
                    print(
                        "Skipping product-target diff norm due to shape mismatch: "
                        f"product={tuple(product_matrix.shape)}, target={tuple(target_matrix.shape)}"
                    )

            for i in range(len(linear_weights) - 1):
                wi = linear_weights[i].detach()
                wi1 = linear_weights[i + 1].detach()

                gram_left = wi @ wi.T
                gram_right = wi1.T @ wi1
                gram_diff = gram_left - gram_right

                logs[f"balance/||W{i}W{i}T||_F"] = torch.linalg.norm(gram_left, ord='fro').item()
                logs[f"balance/||W{i+1}TW{i+1}||_F"] = torch.linalg.norm(gram_right, ord='fro').item()
                logs[f"balance/||W{i} - W{i+1}||_F"] = torch.linalg.norm(gram_diff, ord='fro').item()

        wandb.log(logs)

    def finish(self):
        wandb.finish()