import models
import train
import compressors
from utils import set_seed, load_data, get_device
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
import os
import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":

    trainloader, testloader, classes = load_data()
    device = get_device()

    config = {
        'param_usage': 0.01,
        'num_restarts': 1,
        'num_epochs': 1,
    }

    # list of compressor configurations
    compress_configs = [
        # TopK no error-feedback
        {
            'name': 'TopK',
            'strategy': 'TopK',
            'error_correction': 'none',
            'update_task': None,
            'update_kwargs': {},
            'lr': 0.01,
        },
        # TopK with EF
        {
            'name': 'TopK_EF',
            'strategy': 'TopK',
            'error_correction': 'EF',
            'update_task': None,
            'update_kwargs': {},
            'lr': 0.0025,
        },
        # ImpK mirror descent + EF
        {
            'name': 'ImpK_b_EF',
            'strategy': 'ImpK',
            'error_correction': 'EF',
            'update_task': 'mirror_descent',
            'update_kwargs': {'lambda_value': 0.001, 'start': 'ones'},
            'lr': 0.0025,
            'eta': 1e6,
            'num_steps': 25,
        },
        # SCAM mirror descent + EF
        {
            'name': 'SCAM_b_EF',
            'strategy': 'SCAM',
            'error_correction': 'EF',
            'update_task': 'mirror_descent',
            'update_kwargs': {'lambda_value': 0.001, 'start': 'ones'},
            'lr': 0.01,
            'eta': 1e6,
            'num_steps': 25,
        },
        # ImpK mirror descent no EF
        {
            'name': 'ImpK_b',
            'strategy': 'ImpK',
            'error_correction': 'none',
            'update_task': 'mirror_descent',
            'update_kwargs': {'lambda_value': 0.001, 'start': 'ones'},
            'lr': 0.01,
            'eta': 7.0,
            'num_steps': 25,
        },
        # ImpK gradient descent no EF
        {
            'name': 'ImpK_c',
            'strategy': 'ImpK',
            'error_correction': 'none',
            'update_task': 'gradient_descent',
            'update_kwargs': {'start': 'ones', 'scale': 1.0},
            'lr': 0.01,
            'eta': 1e6,
            'num_steps': 25,
        },
        # ImpK gradient descent + EF
        {
            'name': 'ImpK_c_EF',
            'strategy': 'ImpK',
            'error_correction': 'EF',
            'update_task': 'gradient_descent',
            'update_kwargs': {'start': 'ones', 'scale': 1.0},
            'lr': 0.0025,
            'eta': 1e6,
            'num_steps': 25,
        },
        # SCAM gradient descent + EF
        {
            'name': 'SCAM_c_EF',
            'strategy': 'SCAM',
            'error_correction': 'EF',
            'update_task': 'gradient_descent',
            'update_kwargs': {'start': 'ones', 'scale': 1.0},
            'lr': 0.01,
            'eta': 1e6,
            'num_steps': 25,
        },
        # Full-gradient updates for Impact methods with EF
        {
            'name': 'ImpK_b_EF_full',
            'strategy': 'ImpK',
            'error_correction': 'EF',
            'update_task': 'mirror_descent_full',
            'update_kwargs': {'lambda_value': 0.001, 'start': 'ones'},
            'lr': 0.0025,
            'eta': 1e6,
            'num_steps': 25,
        },
        {
            'name': 'SCAM_b_EF_full',
            'strategy': 'SCAM',
            'error_correction': 'EF',
            'update_task': 'mirror_descent_full',
            'update_kwargs': {'lambda_value': 0.001, 'start': 'ones'},
            'lr': 0.01,
            'eta': 1e6,
            'num_steps': 25,
        },
        {
            'name': 'ImpK_c_EF_full',
            'strategy': 'ImpK',
            'error_correction': 'EF',
            'update_task': 'gradient_descent_full',
            'update_kwargs': {'start': 'ones', 'scale': 1.0},
            'lr': 0.0025,
            'eta': 1e6,
            'num_steps': 25,
        },
        {
            'name': 'SCAM_c_EF_full',
            'strategy': 'SCAM',
            'error_correction': 'EF',
            'update_task': 'gradient_descent_full',
            'update_kwargs': {'start': 'ones', 'scale': 1.0},
            'lr': 0.01,
            'eta': 1e6,
            'num_steps': 25,
        },
    ]


    train_log, train_acc = {}, {}
    test_log, test_acc = {}, {}

    param_usage = config['param_usage']
    num_restarts = config['num_restarts']
    num_epochs = config['num_epochs']

    for compress_config in compress_configs:
        compression_type = compress_config['name']
        # load settings from config
        start       = compress_config.get('start', '')
        lr          = compress_config.get('lr', '')
        eta         = compress_config.get('eta', '')
        num_steps   = compress_config.get('num_steps', '')
        scale       = compress_config.get('scale', '')
        strategy    = compress_config['strategy']
        error_corr  = compress_config['error_correction']
        update_task = compress_config.get('update_task')
        update_kwargs = compress_config.get('update_kwargs', {})

        name = f'{compression_type}_{start}_{lr}'

        train_log[name], train_acc[name], test_log[name], test_acc[name] = [], [], [], []
        
        for num_restart in range(num_restarts):
            set_seed(52 + num_restart)
            net = models.ResNet18().to(device)
            # Use per-config param_usage override if provided
            compressor = compressors.Compressor(
                model=net,
                k=compress_config.get('param_usage', param_usage),
                strategy=strategy,
                error_correction=error_corr,
                update_task=update_task,
                update_kwargs=update_kwargs
            )
            
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            train_loss, train_accuracy, test_loss, test_accuracy = train.train(
                model=net,
                optimizer=optimizer,
                compressor=compressor,
                criterion=criterion,
                train_dataset=trainloader,
                val_dataset=testloader,
                num_epochs=num_epochs,
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                device=device
            )
            print(f"# Compression type: {compression_type}, start: {start}, num_restart: {num_restart}, lr: {lr}, eta: {eta}, num_steps: {num_steps}")
            print("# Train Loss")
            print(train_loss)
            print("# Train Accuracy")
            print(train_accuracy)
            print("# Test Loss")
            print(test_loss)
            print("# Test Accuracy")
            print(test_accuracy)
            train_log[name].append(train_loss)
            train_acc[name].append(train_accuracy)
            test_log[name].append(test_loss)
            test_acc[name].append(test_accuracy)

    print("# Train Loss")
    print(train_log)
    print("# Train Accuracy")
    print(train_acc)
    print("# Test Loss")
    print(test_log)
    print("# Test Accuracy")
    print(test_acc)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{date}.csv")

    with open(log_file, 'w', newline='') as csvfile:
        fieldnames = ['type', 'train_log', 'train_acc', 'test_log', 'test_acc', 'epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for compress_config in compress_configs:
            # use 'name' field for compression type
            compression_type = compress_config['name']
            start = compress_config.get('start', '')
            lr = compress_config.get('lr', '')
            eta = compress_config.get('eta', '')
            num_steps = compress_config.get('num_steps', '')

            name = f'{compression_type}_{param_usage*100:.0f}%_{lr}_{"EF" if "EF" in compression_type else ""}'

            for epoch in range(num_epochs):
                for restart in range(num_restarts):
                    dict_name = f'{compression_type}_{start}_{lr}'
                    writer.writerow({
                        'type': name,
                        'epoch': epoch,
                        'train_log': train_log[dict_name][restart][epoch],
                        'train_acc': train_acc[dict_name][restart][epoch],
                        'test_log': test_log[dict_name][restart][epoch],
                        'test_acc': test_acc[dict_name][restart][epoch]
                    })

    fig_train, axs_train = plt.subplots(1, 2, figsize=(16, 7))
    fig_test, axs_test = plt.subplots(1, 2, figsize=(16, 7))


    for compress_config in compress_configs:
        # use 'name' for compression type
        compression_type = compress_config['name']

        start = compress_config.get('start', '')
        lr = compress_config.get('lr', '')
        eta = compress_config.get('eta', '')
        num_steps = compress_config.get('num_steps', '')

        name = f'{compression_type}_{start}_{lr}'

        train_loss = np.array(train_log[name])
        train_loss_mean = np.mean(train_loss, axis=0)
        train_loss_std = np.std(train_loss, axis=0)
        
        train_accuracy = np.array(train_acc[name])
        train_accuracy_mean = np.mean(train_accuracy, axis=0)
        train_accuracy_std = np.std(train_accuracy, axis=0)
        
        test_loss = np.array(test_log[name])
        test_loss_mean = np.mean(test_loss, axis=0)
        test_loss_std = np.std(test_loss, axis=0)
        
        test_accuracy = np.array(test_acc[name])
        test_accuracy_mean = np.mean(test_accuracy, axis=0)
        test_accuracy_std = np.std(test_accuracy, axis=0)
        
        iters = list(range(len(train_loss_mean)))
        
        axs_train[0].plot(iters, train_loss_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_train[0].fill_between(iters, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1)
        
        axs_train[1].plot(iters, train_accuracy_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_train[1].fill_between(iters, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1)

        axs_test[0].plot(iters, test_loss_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_test[0].fill_between(iters, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.1)
        
        axs_test[1].plot(iters, test_accuracy_mean, label=f'{compression_type}, lr={lr}, start={start}')
        axs_test[1].fill_between(iters, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1)

    axs_train[0].set_title(f"Comparison on Train, different compression types, param_usage={param_usage}")
    axs_train[0].set_xlabel("Epoch")
    axs_train[0].set_ylabel("Loss")
    axs_train[0].legend()
    axs_train[0].grid()

    axs_train[1].set_title(f"Comparison on Train, different compression types, param_usage={param_usage}")
    axs_train[1].set_xlabel("Epoch")
    axs_train[1].set_ylabel("Accuracy")
    axs_train[1].legend()
    axs_train[1].grid()
        

    axs_test[0].set_title(f"Comparison on Test, different compression types, param_usage={param_usage}")
    axs_test[0].set_xlabel("Epoch")
    axs_test[0].set_ylabel("Loss")
    axs_test[0].legend()
    axs_test[0].grid()

    axs_test[1].set_title(f"Comparison on Test, different compression types, param_usage={param_usage}")
    axs_test[1].set_xlabel("Epoch")
    axs_test[1].set_ylabel("Accuracy")
    axs_test[1].legend()
    axs_test[1].grid()

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if the directory 'figures' exists, if not, create it
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save the train plot in the 'figures' directory
    fig_train.savefig(os.path.join(figures_dir, f"train_comparison_param_usage_{param_usage}_{date}.png"))

    # Save the test plot in the 'figures' directory
    fig_test.savefig(os.path.join(figures_dir, f"test_comparison_param_usage_{param_usage}_{date}.png"))

    fig_train.show()
    fig_test.show()