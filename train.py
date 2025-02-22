import torch
from tqdm import tqdm, trange

def train(model, optimizer, compressor, criterion, train_dataset, val_dataset, num_epochs, lr, eta, num_steps, device, quiet=False):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []
    for epoch in trange(num_epochs):
        if not quiet:
            tqdm.write('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
              
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            inputs, targets = inputs.to(device), targets.to(device)

            if batch_idx == 0:
                compressor.update(inputs, targets, criterion, lr, eta, num_steps)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if 'bn' in name or 'shortcut.1' in name:
                    continue
                param.grad.copy_(compressor.compress(name, param))
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_dataset)
        train_accuracy = 100. * correct / total
        train_log.append(train_loss)
        train_acc_log.append(train_accuracy)
        
        if not quiet:
            tqdm.write('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss, train_accuracy, correct, total))
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_dataset:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_dataset)
        val_accuracy = 100. * correct / total
        val_log.append(val_loss)
        val_acc_log.append(val_accuracy)
        if not quiet:
            tqdm.write('\nValidation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)' % (val_loss, val_accuracy, correct, total))
    
    return train_log, train_acc_log, val_log, val_acc_log