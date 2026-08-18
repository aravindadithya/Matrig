import os
import time

import torch
import wandb
import torch.nn.functional as F

from utils.base_logger import BaseLogger


def compute_dfa_task_signal(net, inputs, target, lfn):
    """Compute the global output error used by DFA.

    This is the key difference from RFA: every hidden layer receives the same
    network-level error signal, not the local downstream gradient.
    """
    net.train()
    output = net(inputs)
    loss = lfn(output, target)
    grad_output = torch.autograd.grad(loss, output, retain_graph=False, create_graph=False)[0]
    return grad_output.detach()


def train_step_dfa(net, optimizer, lfn, train_loader, config):
    """Train a DFA network with the global output error injected into each layer."""
    device = next(net.parameters()).device
    net.train()
    start = time.time()

    train_loss_accum = torch.tensor(0.0, device=device)
    correct_accum = torch.tensor(0.0, device=device)
    total = 0
    non_critical = os.getenv('NON_CRITICAL_LOGS', 'False').lower() in ('true', '1', 't')

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        inputs, labels = batch

        inputs = inputs.to(device=device, non_blocking=True)
        target = labels.to(device=device, non_blocking=True)

        # Build the global output error first; this is the DFA signal used for all layers.
        with torch.enable_grad():
            proxy_output = net(inputs)
            proxy_loss = lfn(proxy_output, target)
            task_signal = torch.autograd.grad(proxy_loss, proxy_output, retain_graph=False, create_graph=False)[0]

        output = net(inputs, task_signal=task_signal)
        loss = lfn(output, target)

        loss.backward()
        optimizer.step()

        train_loss_accum += loss.detach() * inputs.size(0)

        if batch_idx % 10 == 0 and non_critical:
            wandb.log({"Batch/loss": loss.item()})

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        if len(target.size()) > 1:
            _, labels_idx = torch.max(target, -1)
        else:
            labels_idx = target
        correct_accum += (predicted == labels_idx).sum()

    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss_accum.item() / len(train_loader.dataset)
    train_acc = 100 * correct_accum.item() / total
    return train_loss, train_acc


def val_step_dfa(net, val_loader, config, lfn=None):
    """Validation pass for DFA network."""
    device = next(net.parameters()).device
    net.eval()
    val_loss_accum = torch.tensor(0.0, device=device)
    correct_accum = torch.tensor(0.0, device=device)
    total = 0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        inputs = inputs.to(device=device, non_blocking=True)
        target = labels.to(device=device, non_blocking=True)

        with torch.no_grad():
            output = net(inputs)
            if lfn:
                loss = lfn(output, target)
                val_loss_accum += loss.detach() * inputs.size(0)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            if len(target.size()) > 1:
                _, labels_idx = torch.max(target, -1)
            else:
                labels_idx = target

            correct_accum += (predicted == labels_idx).sum()
            all_preds.append(predicted)
            all_targets.append(labels_idx)

    if lfn:
        val_loss = val_loss_accum.item() / len(val_loader.dataset)
    else:
        val_loss = None

    val_acc = 100 * correct_accum.item() / total
    all_preds = torch.cat(all_preds).cpu().tolist()
    all_targets = torch.cat(all_targets).cpu().tolist()
    return val_loss, val_acc, all_preds, all_targets


def train_network_dfa(config, num_epochs=5, checkpoint_interval=10):
    """Run a separate DFA training loop without modifying the standard trainer."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = config['net']
    optimizer = config['optimizer']
    lfn = config['lfn']
    scheduler = config.get('scheduler')
    train_loader = config['train_loader']
    val_loader = config['val_loader']
    test_loader = config['test_loader']

    net.cuda()

    print("Initializing Wandb:")
    logger = BaseLogger(config)
    wandb.watch(net, log="all", log_freq=100, idx=0)

    best_test_acc = 0
    best_test_loss = 0
    start_epoch = logger.start_epoch
    best_val_acc = logger.best_val_acc
    best_val_loss = logger.best_val_loss
    best_state_dict = logger.best_state_dict

    for i in range(start_epoch, start_epoch + num_epochs):
        print("EPOCH: ", i)
        logger.log_matrix_diagnostics(i)

        train_loss_full, train_acc_full, train_preds, train_targets = val_step_dfa(net, train_loader, config, lfn)
        val_loss, val_acc, val_preds, val_targets = val_step_dfa(net, val_loader, config, lfn)

        train_loss, train_acc = train_step_dfa(net, optimizer, lfn, train_loader, config)

        log_data = {
            "epoch": i,
            "train/accuracy": train_acc_full,
            "train/loss": train_loss_full,
            "val/accuracy": val_acc,
            "val/loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                try:
                    scheduler.step()
                except ValueError as e:
                    if "Tried to step" in str(e):
                        print(f"Scheduler finished: {e}. Stopping training.")
                        break
                    raise e

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state_dict = net.state_dict()
            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_val_loss"] = best_val_loss

            if logger.inputs is not None:
                logger.log_visuals(net, epoch=i)

        if i % checkpoint_interval == 0:
            artifact = wandb.Artifact(f"checkpoint-{config['run_id']}", type='model', metadata={"val_acc": val_acc, "best_val_acc": best_val_acc, "epoch": i})
            with artifact.new_file('last_model.pth', mode='wb') as f:
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None
                }, f)
            wandb.log_artifact(artifact)

        wandb.log(log_data)

    if best_state_dict:
        artifact = wandb.Artifact(f"model-{config['run_id']}", type='model', metadata={"best_val_acc": best_val_acc})
        with artifact.new_file('best_model.pth', mode='wb') as f:
            torch.save({'state_dict': best_state_dict}, f)
        wandb.log_artifact(artifact)
        net.load_state_dict(best_state_dict)

    best_test_loss, best_test_acc, test_preds, test_targets = val_step_dfa(net, test_loader, config, lfn)
    wandb.run.summary["best_test_accuracy"] = best_test_acc
    wandb.run.summary["best_test_loss"] = best_test_loss

    logger.finish()
    print("FINISHED DFA TRAINING :)")
