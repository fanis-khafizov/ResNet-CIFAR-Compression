import os
from dotenv import load_dotenv

import wandb
import torch
from utils import load_data, get_device
from config import ExperimentConfig
from optimizers import CSGD, CAdamW
from experiment import Experiment

# Load environment variables
load_dotenv()

# Authenticate with W&B using API key from environment variable
wandb.login(key=os.environ.get("WANDB_API_KEY"))

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    trainloader, testloader, classes = load_data()
    device = get_device()

    train_config = {
        'param_usage': 0.01,
        'num_restarts': 5,
        'num_epochs': 50,
    }
    # Извлечение настроек из train_config
    param_usage = train_config['param_usage']
    num_restarts = train_config['num_restarts']
    num_epochs = train_config['num_epochs']

    configs = [
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment2', name='TopK_0.001', strategy='TopK', lr=0.001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment2', name='TopK_EF_0.002', strategy='TopK', error_correction='EF', lr=0.002, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment2', name='TopK_EF21_0.001', strategy='TopK', error_correction='EF21', lr=0.001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment2', name='SCAM_TopK_0.0001', strategy='TopK', error_correction='EF', lr=0.0001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment2', name='SCAM_c_EF_0.0001', strategy='SCAM', error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e6,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
    ]
    for cfg in configs:
        experiment = Experiment(cfg, trainloader, testloader, device, param_usage, num_epochs, num_restarts)
        experiment.run()