import torch
from tqdm import tqdm, trange
import time

import wandb

def train(model, criterion, optimizer, compressor, trainloader, testloader, num_epochs, device, update_freq=1):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in trange(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Measure compressor update time
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            if batch_idx == 0 and epoch % update_freq == 0:  # Update only if condition is met
                compressor.update(inputs, targets, criterion)
                update_time = time.time() - start_time
                wandb.log({"train/update_time": update_time}, step=epoch)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= len(trainloader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Measure train epoch time
        train_epoch_time = time.time() - start_time
        wandb.log({"train/epoch_time": train_epoch_time}, step=epoch)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_start_time = time.time()
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(testloader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Measure validation epoch time
        val_epoch_time = time.time() - val_start_time
        wandb.log({"val/epoch_time": val_epoch_time}, step=epoch)

        # Log metrics and epoch time
        wandb.log({
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc
        }, step=epoch)

    return train_losses, train_accs, val_losses, val_accs