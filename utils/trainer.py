
import torch
import wandb
import time
import os
# from torch.amp import autocast
import torch.nn.functional as F
from utils.base_logger import BaseLogger
# scaler = torch.amp.GradScaler('cuda')
fn_data = {}


def train_network(config, num_epochs = 5, checkpoint_interval=10):

    # Force full FP32 for bit-perfect parity between native and custom layers
    torch.set_float32_matmul_precision('high')

    # For full reproducibility
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

    #net = torch.compile(net)

    print("Initializing Wandb:")
    logger = BaseLogger(config)

    
    # wandb.watch can cause significant overhead or hangs with log="all" on some systems
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
        #logger.log_singular_values(i)
        #logger.log_agop(i)
        #logger.count_sparsity(i)

        #train_loss_full, train_acc_full , train_preds, train_targets= val_step(net, train_loader, config, lfn)
        val_loss, val_acc = val_step(net, val_loader, config, lfn)

        train_loss, train_acc = train_step(net, optimizer, lfn, train_loader, config)
        # Validation loss and accuracy are calculated after backprob for each epoch
        
        log_data = {
            "epoch": i,
            "train/accuracy": train_acc,
            "train/loss": train_loss,
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

    best_test_loss, best_test_acc = val_step(net, test_loader, config, lfn)
    wandb.run.summary["best_test_accuracy"] = best_test_acc
    wandb.run.summary["best_test_loss"] = best_test_loss
    

    logger.finish()
    
    print("FINISHED TRAINING :)")


def train_step(net, optimizer, lfn, train_loader, config):
    # global scaler
    net.train()
    start = time.time()
    # Accumulate on GPU to avoid CPU-GPU sync in the loop
    train_loss_accum = torch.tensor(0.0, device='cuda')
    correct_accum = torch.tensor(0.0, device='cuda')
    total = 0
    non_critical = os.getenv('NON_CRITICAL_LOGS', 'False').lower() in ('true', '1', 't')

    memory_format = torch.channels_last if config.get('memory_format') == 'channels_last' else torch.contiguous_format
    for batch_idx, batch in enumerate(train_loader):
        # Optimization: set_to_none=True skips zeroing the memory, which is faster
        optimizer.zero_grad(set_to_none=True)
        inputs, labels = batch
        targets = labels

        inputs = inputs.to(device='cuda', non_blocking=True)
        target = targets.cuda(non_blocking=True)

        # with autocast(device_type='cuda'):
        #     output = net(inputs)
        #     loss = lfn(output, target)
        output = net(inputs)
        loss = lfn(output, target)
        
        # scaler.scale(loss).backward()  
        loss.backward()
        
        # scaler.step(optimizer)    
        # scaler.update() 
        optimizer.step()

        train_loss_accum += loss.detach() * inputs.size(0)
        # Note: loss.item() triggers a CPU-GPU sync
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


def val_step(net, val_loader, config, lfn=None, return_predictions=False):
    net.eval()
    val_loss_accum = torch.tensor(0.0, device='cuda')
    correct_accum = torch.tensor(0.0, device='cuda')
    total = 0
    
    all_preds = [] if return_predictions else None
    all_targets = [] if return_predictions else None

    memory_format = torch.channels_last if config.get('memory_format') == 'channels_last' else torch.contiguous_format

    with torch.no_grad():
        for inputs, targets in val_loader:
            if inputs.dim() == 4:
                inputs = inputs.to(device='cuda', memory_format=memory_format, non_blocking=True)
            else:
                inputs = inputs.to(device='cuda', non_blocking=True)
            
            target = targets.to(device='cuda', non_blocking=True)

            output = net(inputs)
            if lfn:
                loss = lfn(output, target)
                val_loss_accum += loss.detach() * inputs.size(0)

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            labels_idx = torch.max(target, -1)[1] if target.dim() > 1 else target

            correct_accum += (predicted == labels_idx).sum()

            if return_predictions:
                all_preds.append(predicted)
                all_targets.append(labels_idx)

    val_loss = (val_loss_accum.item() / total) if lfn else None
    val_acc = 100.0 * correct_accum.item() / total

    if return_predictions:
        all_preds = torch.cat(all_preds).cpu().tolist()
        all_targets = torch.cat(all_targets).cpu().tolist()
        return val_loss, val_acc, all_preds, all_targets

    return val_loss, val_acc


def get_trained_net(config):
    net = config['net']
    api = wandb.Api()
    try:
        artifact = api.artifact(f"{config['entity']}/{config['project']}/model-{config['run_id']}:latest")
        model_dir = artifact.download()
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'), weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights from artifact: {artifact.name}")
        
    except Exception as e:
        print(f"Error loading from WandB: {e}")
    return net

def cleanup_artifacts(config):
    api = wandb.Api()
    try:
        versions = api.artifact_versions("model", f"{config['entity']}/{config['project']}/model-{config['run_id']}")
        for v in versions:
            if 'latest' not in v.aliases:
                v.delete()
        versions = api.artifact_versions("model", f"{config['entity']}/{config['project']}/checkpoint-{config['run_id']}")
        for v in versions:
            v.delete()

    except Exception:
        print("Error cleaning up artifacts")
        pass



'''
def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

'''