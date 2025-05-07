import os
import csv
import datetime
import wandb

class Logger:
    def __init__(self, name: str, param_usage: float):
        self.name = name
        self.param_usage = param_usage
        self.records = []  # список словарей: {'epoch', 'restart', 'train_loss', 'train_acc', 'val_loss', 'val_acc'}

    def log(self, epoch: int, restart: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        # Логирование в W&B
        wandb.log({
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc
        }, step=epoch)
        # Сохранение записи локально
        self.records.append({
            'epoch': epoch,
            'restart': restart,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    def save_csv(self, log_dir: str = 'logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.name}_{self.param_usage*100:.0f}%_{date}.csv"
        path = os.path.join(log_dir, fname)
        fieldnames = ['epoch', 'restart', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec)

    def plot(self, plot_func):
        # Группируем по рестартам и эпохам
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        # Преобразуем записи в структуру: списки по рестартам
        restarts = max(rec['restart'] for rec in self.records) + 1 if self.records else 0
        epochs = max(rec['epoch'] for rec in self.records) + 1 if self.records else 0
        # Инициализация
        train_losses = [[0.0]*epochs for _ in range(restarts)]
        train_accs = [[0.0]*epochs for _ in range(restarts)]
        val_losses = [[0.0]*epochs for _ in range(restarts)]
        val_accs = [[0.0]*epochs for _ in range(restarts)]
        for rec in self.records:
            r = rec['restart']
            e = rec['epoch']
            train_losses[r][e] = rec['train_loss']
            train_accs[r][e] = rec['train_acc']
            val_losses[r][e] = rec['val_loss']
            val_accs[r][e] = rec['val_acc']
        # Вызов функции построения графиков
        plot_func(train_losses, train_accs, val_losses, val_accs, self.name)
