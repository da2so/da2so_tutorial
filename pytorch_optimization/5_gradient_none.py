'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from loguru import logger
from pathlib import Path

from models import *
from utils import progress_bar, setup_logger, de_parallel



class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.gn = args.gradient_none
        num_workers, pin_memory, self.epochs = args.num_workers, args.pin_memory, args.epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_time = 0.
        self.test_time = 0.
        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')


        # Model
        print('==> Building model..')
        self.net = model
        self.net = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.log_path = setup_logger(save_dir='results', filename=f'{Path(__file__).stem}.log', mode="a")

    def run(self):
        for epoch in range(self.epochs):
            train_start = time.time()
            self.train(epoch)
            train_end = time.time()
            test_start = time.time()
            self.test(epoch)
            test_end = time.time()

            self.scheduler.step()

            if epoch >= 2:
                self.train_time += train_end-train_start
                self.test_time += test_end-test_start
        logger.info(f'Model: {de_parallel(self.net).name}')
        logger.info(f'config: {vars(args)}')
        logger.info(f'Avg train time elapse: {(self.train_time/(self.epochs-2)):.3f}')
        logger.info(f'Avg test time elapse: {(self.test_time/(self.epochs-2)):.3f}\n')

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.gn:
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad()

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='pin memory')
    parser.add_argument('--gradient_none', default=True, type=bool, help='set gradient to None')
    parser.add_argument('--epochs', default=5, type=int, help='epochs')
    args = parser.parse_args()
    
    
    gradient_none = [False, True]
    model = [ResNet18(), ResNet50(), ResNet101(), MobileNetV2()]
    for m in model:
        for gn in gradient_none:
            args.gradient_none = gn 
            trainer = Trainer(args=args, model=m)
            trainer.run()
