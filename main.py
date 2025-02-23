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
    print(device)

    config = {
        'param_usage': 0.005,
        'num_restarts': 1,
        'num_epochs': 50,
    }

    compress_configs = [
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.05,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.075,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.01,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.005,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.0025,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.001,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.00075,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.01,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.02,
        # },
        # {
        #     'compression_type': 'TopK',
        #     'lr': 0.05,
        # },
        # {
        #     'compression_type': 'TopK_EF',
        #     'lr': 0.01,
        # },
        # {
        #     'compression_type': 'ImpK_b_EF',
        #     'start': 'ones',
        #     'lr': 0.001,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        # },
        # {
        #     'compression_type': 'SCAM_b_EF',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        # },
        {
            'compression_type': 'ImpK_b',
            'start': 'ones',
            'lr': 0.01,
            'eta': 1000000.,
            'num_steps': 25,
        },
        {
            'compression_type': 'ImpK_b',
            'start': 'ones',
            'lr': 0.005,
            'eta': 1000000.,
            'num_steps': 25,
        },
        {
            'compression_type': 'ImpK_b',
            'start': 'ones',
            'lr': 0.0025,
            'eta': 1000000.,
            'num_steps': 25,
        },
        {
            'compression_type': 'ImpK_b',
            'start': 'ones',
            'lr': 0.001,
            'eta': 1000000.,
            'num_steps': 25,
        },
        # {
        #     'compression_type': 'ImpK_c_EF21',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'ImpK_c',
        #     'start': 'ones',
        #     'lr': 0.015,
        #     'eta': 1000000.,
        #     'num_steps': 20,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'ImpK_c',
        #     'start': 'ones',
        #     'lr': 0.02,
        #     'eta': 1000000.,
        #     'num_steps': 20,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'ImpK_c_EF',
        #     'start': 'ones',
        #     'lr': 0.001,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        #     'scale': 1.0,
        # },
        # {
        #     'compression_type': 'SCAM_c_EF',
        #     'start': 'ones',
        #     'lr': 0.01,
        #     'eta': 1000000.,
        #     'num_steps': 25,
        #     'scale': 1.0,
        # },
    ]


    train_log, train_acc = {}, {}
    test_log, test_acc = {}, {}

    param_usage = config['param_usage']
    num_restarts = config['num_restarts']
    num_epochs = config['num_epochs']

    for compress_config in compress_configs:
        compression_type = compress_config['compression_type']

        start = compress_config.get('start', '')
        lr = compress_config.get('lr', '')
        eta = compress_config.get('eta', '')
        num_steps = compress_config.get('num_steps', '')
        scale=compress_config.get('scale', '')

        name = f'{compression_type}_{start}_{lr}'

        train_log[name], train_acc[name], test_log[name], test_acc[name] = [], [], [], []
        
        for num_restart in range(num_restarts):
            set_seed(52 + num_restart)
            net = models.ResNet18().to(device)

            if compression_type == 'TopK':
                compressor = compressors.TopK(param_usage)
            elif compression_type == 'TopK_EF':
                compressor = compressors.TopK_EF(param_usage, net)
            elif compression_type == 'TopK_EF21':
                compressor = compressors.TopK_EF21(param_usage, net)
            elif compression_type == 'RandK':
                compressor = compressors.RandK(param_usage)
            elif compression_type == 'ImpK_b':
                compressor = compressors.ImpK_b(net, param_usage, start=start)
            elif compression_type == 'ImpK_b_EF':
                compressor = compressors.ImpK_b_EF(net, param_usage, start=start)
            elif compression_type == 'ImpK_b_EF21':
                compressor = compressors.ImpK_b_EF21(net, param_usage, start=start)
            elif compression_type == 'ImpK_c':
                compressor = compressors.ImpK_c(net, param_usage, start=start, scale=scale)
            elif compression_type == 'ImpK_c_EF':
                compressor = compressors.ImpK_c_EF(net, param_usage, start=start, scale=scale)
            elif compression_type == 'ImpK_c_EF21':
                compressor = compressors.ImpK_c_EF21(net, param_usage, start=start, scale=scale)
            elif compression_type == 'SCAM_b_EF':
                compressor = compressors.SCAM_b_EF(net, param_usage, start=start)
            elif compression_type == 'SCAM_c_EF':
                compressor = compressors.SCAM_c_EF(net, param_usage, start=start, scale=scale)
            else:
                raise ValueError(f"Unknown compression type: {compression_type}")
            
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

        if '_c_' not in compression_type:
            filename = f"{compression_type}_{int(100 * param_usage)}%_{lr}.txt"
        else:
            filename = f"{compression_type}_[0,{int(scale)}]_{int(1000 * param_usage) / 10}%_{lr}.txt"
        with open(filename, "w") as file:
            file.write(f'{train_log=}, {train_acc=}, {test_log=}, {test_acc=}')