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
        'num_restarts': 1,
        'num_epochs': 50,
    }
    # Извлечение настроек из train_config
    param_usage = train_config['param_usage']
    num_restarts = train_config['num_restarts']
    num_epochs = train_config['num_epochs']

    configs = [
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='ImpK_c_EF_0.002_1', strategy='ImpK', update_freq=1, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.002, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='ImpK_c_EF_0.002_2', strategy='ImpK', update_freq=2, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.002, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='ImpK_c_EF_0.002_4', strategy='ImpK', update_freq=4, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.002, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='ImpK_c_EF_0.002_8', strategy='ImpK', update_freq=8, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.002, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='ImpK_c_EF_0.002_16', strategy='ImpK', update_freq=16, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e7,'num_steps':50}, lr=0.002, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='SCAM_c_EF_0.0001_1', strategy='SCAM', update_freq=1, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e6,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='SCAM_c_EF_0.0001_2', strategy='SCAM', update_freq=2, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e6,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='SCAM_c_EF_0.0001_4', strategy='SCAM', update_freq=4, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e6,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='SCAM_c_EF_0.0001_8', strategy='SCAM', update_freq=8, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e6,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
        ExperimentConfig(train_config, project_name='ResNet-CIFAR-Experiment3', name='SCAM_c_EF_0.0001_16', strategy='SCAM', update_freq=16, error_correction='EF', update_task='gradient_descent', update_kwargs={'scale':2.0,'eta':1e6,'num_steps':50}, lr=0.0001, optimizer=CAdamW),
    ]
    for cfg in configs:
        experiment = Experiment(cfg, trainloader, testloader, device, param_usage, num_epochs, num_restarts)
        experiment.run()