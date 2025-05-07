import torch
from tqdm import tqdm, trange

def train(model, criterion, optimizer, compressor, trainloader, testloader, num_epochs, lr, eta, num_steps, device, logger, restart=0, quiet=True):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in trange(num_epochs):
        if not quiet:
            tqdm.write('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
              
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)

            if batch_idx == 0:
                compressor.update(inputs, targets, criterion, lr, eta, num_steps)
            
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
        

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
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
        
        logger.log(epoch, restart, train_loss, train_acc, val_loss, val_acc)
        
        if not quiet:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            print(f"Epoch {epoch+1}, Train PPL: {train_acc}, Val PPL: {val_acc}")
    
    
    return train_losses, train_accs, val_losses, val_accs