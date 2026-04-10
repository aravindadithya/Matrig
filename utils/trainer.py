
import torch
import wandb
import time
import os
from torch.amp import autocast
import torch.nn.functional as F
from utils.base_logger import BaseLogger
scaler = torch.amp.GradScaler('cuda')
fn_data = {}


def train_network(config, num_epochs = 5, checkpoint_interval=10):

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
    # This is supposed to improve performance but I found mixed results 
    # with diminishing returns
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
        logger.log_agop(i)
        logger.count_sparsity(i)
        
        #Train loss and accuracy are calculated on the fly during backprob for each epoch
        train_loss, train_acc = train_step(net, optimizer, lfn, train_loader, config)
        # Validation loss and accuracy are calculated after backprob for each epoch
        val_loss, val_acc, val_preds, val_targets = val_step(net, val_loader, config, lfn)       

        
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
            
            # logger.log_confusion_matrix(y_true=val_targets, preds=val_preds,
            #                             epoch=i, class_names=[str(c) for c in range(10)],
            #                             log_key="Validation Confusion Matrix")

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

    best_test_loss, best_test_acc, test_preds, test_targets = val_step(net, test_loader, config, lfn)
    wandb.run.summary["best_test_accuracy"] = best_test_acc
    wandb.run.summary["best_test_loss"] = best_test_loss
    
    # logger.log_confusion_matrix(y_true=test_targets, preds=test_preds,
    #                             epoch=i, class_names=[str(i) for i in range(10)],
    #                             log_key="Test Confusion Matrix")

    #logger.log_predictions_table(net, test_loader, epoch=i, log_key="Test Predictions", limit=256)
    

    logger.finish()
    
    print("FINISHED TRAINING :)")


def train_step(net, optimizer, lfn, train_loader, config):
    global scaler
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
        
        # Optimization: Channels Last for inputs matches the model layout
        if inputs.dim() == 4:
            inputs = inputs.to(device='cuda', memory_format=memory_format, non_blocking=True)
        else:
            inputs = inputs.to(device='cuda', non_blocking=True)
        target = targets.cuda(non_blocking=True)

        with autocast(device_type='cuda'):
            output = net(inputs)
            loss = lfn(output, target)
        
        scaler.scale(loss).backward()  
        
        scaler.step(optimizer)    
        scaler.update() 

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


def val_step(net, val_loader, config, lfn=None):
    global scaler
    net.eval()
    val_loss_accum = torch.tensor(0.0, device='cuda')
    correct_accum = torch.tensor(0.0, device='cuda')
    total = 0
    all_preds = []
    all_targets = []
    val_loss = None

    memory_format = torch.channels_last if config.get('memory_format') == 'channels_last' else torch.contiguous_format
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        if inputs.dim() == 4:
            inputs = inputs.to(device='cuda', memory_format=memory_format, non_blocking=True)
        else:
            inputs = inputs.to(device='cuda', non_blocking=True)
        target = targets.cuda(non_blocking=True)

        with torch.no_grad():
            with autocast(device_type='cuda'): 
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

    val_acc = 100 * correct_accum.item() / total
    all_preds = torch.cat(all_preds).cpu().tolist()
    all_targets = torch.cat(all_targets).cpu().tolist()

    return val_loss, val_acc, all_preds, all_targets


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
