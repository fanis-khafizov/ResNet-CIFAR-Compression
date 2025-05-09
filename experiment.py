import wandb
import torch
from torch.nn import CrossEntropyLoss
from models import ResNet18

from train import train
from utils import set_seed
import compressors

class Experiment:
    def __init__(self, config, trainloader, testloader, device, param_usage, num_epochs, num_restarts):
        self.config = config
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.param_usage = param_usage
        self.num_epochs = num_epochs
        self.num_restarts = num_restarts

    def run(self):
        for restart in range(self.num_restarts):
            # Initialize W&B for each restart
            wandb.init(
                project=self.config.project_name,
                name=f"{self.config.name}_restart_{restart}",
                config=self.config.__dict__
            )

            set_seed(52 + restart)

            # Create ResNet18 model
            model = ResNet18().to(self.device)

            # Set up criterion and compressor
            criterion = CrossEntropyLoss()
            compressor = compressors.Compressor(
                model=model,
                k=self.param_usage,
                strategy=self.config.strategy,
                error_correction=self.config.error_correction,
                update_task=self.config.update_task,
                lr=self.config.lr,
                update_kwargs=self.config.update_kwargs
            )

            # Instantiate optimizer from config
            optimizer = self.config.optimizer(
                compressor=compressor,
                lr=self.config.lr,
                **self.config.optimizer_kwargs
            )

            # Training loop with logging
            train(
                model=model,
                config=self.config,
                criterion=criterion,
                optimizer=optimizer,
                compressor=compressor,
                trainloader=self.trainloader,
                testloader=self.testloader,
                num_epochs=self.num_epochs,
                device=self.device,
            )

            # Finish W&B for the current restart
            wandb.finish()
