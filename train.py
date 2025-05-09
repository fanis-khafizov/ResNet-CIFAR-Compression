import torch
from tqdm import tqdm, trange
import time

import wandb

def train(model, config, criterion, optimizer, compressor, trainloader, testloader, num_epochs, device, quiet=True):
    lr, eta, num_steps = config.lr, config.eta, config.num_steps

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in trange(num_epochs):

        if not quiet:
            tqdm.write('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Measure compressor update time
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            if batch_idx == 0:
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
        config.log(train_loss, train_acc, val_loss, val_acc, epoch)

        if not quiet:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            print(f"Epoch {epoch+1}, Train PPL: {train_acc}, Val PPL: {val_acc}")
    
    
    return train_losses, train_accs, val_losses, val_accs