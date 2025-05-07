import wandb
import torch
from torch.nn import CrossEntropyLoss
from models import ResNet18

from logger import Logger
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
        self.logger = Logger(config.name, param_usage)

    def run(self):
        # Initialize W&B
        self.config.init_wandb()

        for restart in range(self.num_restarts):
            set_seed(52 + restart)

            # Create ResNet18 model
            model = ResNet18().to(self.device)
            model = torch.compile(model)

            # Set up criterion and compressor
            criterion = CrossEntropyLoss()
            compressor = compressors.Compressor(
                model=model,
                k=self.param_usage,
                strategy=self.config.strategy,
                error_correction=self.config.error_correction,
                update_task=self.config.update_task,
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
                criterion=criterion,
                optimizer=optimizer,
                compressor=compressor,
                trainloader=self.trainloader,
                testloader=self.testloader,
                num_epochs=self.num_epochs,
                lr=self.config.lr,
                eta=self.config.eta,
                num_steps=self.config.num_steps,
                device=self.device,
                logger=self.logger,
                restart=restart
            )

        # Finish W&B
        wandb.finish()
        # Save results
        self.logger.save_csv()
