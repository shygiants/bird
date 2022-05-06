import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
import torchmetrics


class Birds400(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, shuffle=True, num_workers=1, pin_memory=False):
        super().__init__()
        self.save_hyperparameters('batch_size', 'shuffle')
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_classes = 400

    @staticmethod
    def build_transform(augment):
        ts = [
            transforms.ToTensor(),
        ]

        target_size = (224, 224)

        if augment:
            ts.append(transforms.RandomHorizontalFlip())
            ts.append(transforms.RandomResizedCrop(target_size))
        else:
            ts.append(transforms.Resize(target_size))

        return transforms.Compose(ts)

    def load_dataset(self, split, augment=None):
        if augment is None:
            augment = split == 'train'

        split_dir = os.path.join(self.data_dir, split)
        transform = Birds400.build_transform(augment)
        return ImageFolder(split_dir, transform=transform)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.ds_train = self.load_dataset('train')
            self.ds_val = self.load_dataset('valid')
            self.ds_train_eval = self.load_dataset('train', augment=False)
        if stage == 'test' or stage is None:
            self.ds_test = self.load_dataset('test')
        if stage == 'predict' or stage is None:
            self.ds_predict = self.load_dataset('test')

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.hparams.batch_size,
                          shuffle=self.hparams.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return [
            DataLoader(
                self.ds_val,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            ), DataLoader(
                self.ds_train_eval,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        ]

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class MobileNetV3Small(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters('learning_rate')
        self.net = mobilenet_v3_small(num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)

        loss = self.criterion(logit, y)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        logit = self(x)

        split = 'val' if dataloader_idx == 0 else 'train'
        acc = self.val_acc if dataloader_idx == 0 else self.train_acc
        acc(logit, y)

        self.log(f'{split}_acc', acc, add_dataloader_idx=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)

        self.val_acc(logit, y)

        self.log('test_acc', self.val_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


cli = LightningCLI(
    MobileNetV3Small,
    Birds400,
    parser_kwargs={
        'fit': {
            'default_config_files': [
                'config/config.yaml'
            ]
        }
    },
)
