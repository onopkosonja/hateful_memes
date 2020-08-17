from byol_pytorch import BYOL

from PIL import Image
from pathlib import Path
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import os
import json
import torch
import neptune
import multiprocessing
import pytorch_lightning as pl


class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.log_metric('train_loss', avg_loss)
        return {'train_loss': avg_loss}

    def validation_step(self, images, _):
        loss = self.forward(images)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
         self.logger.experiment.log_metric('val_loss', avg_loss)
         return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()


def expand_greyscale(t):
    return t.expand(3, -1, -1)


class ImagesDataset(Dataset):
    def __init__(self, json_path, folder, image_size):
        super().__init__()
        self.folder = folder

        with open(json_path, 'r') as file:
            dataset = file.readlines()

        self.dataset = [json.load(l) for l in dataset]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.dataset[index]['img'])
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)


if __name__ == '__main__':
    RESNET       = models.resnet50(pretrained=False)
    BATCH_SIZE   = 32
    EPOCHS       = 1000
    LR           = 3e-4
    NUM_GPUS     = 1
    IMAGE_SIZE   = 256
    IMAGE_EXTS   = ['.png']
    NUM_WORKERS  = multiprocessing.cpu_count()

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    TRAIN_DATASET = ImagesDataset('data/train.jsonl', 'data/img', IMAGE_SIZE)
    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    VAL_DATASET = ImagesDataset('data/dev.jsonl', 'data/img', IMAGE_SIZE)
    VAL_LOADER = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    neptune.init(
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNWYyMzI4ZTYtYmNhYy00MTVjLTg3ZTQtMGJhMzRkNmNiNTBiIn0=',
        project_qualified_name='onopkosonja/byol')

    model = SelfSupervisedLearner(
        RESNET,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99)

    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/resnet50_not_pretrained' + '_{epoch}',
        save_top_k=5, monitor='val_loss')

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNWYyMzI4ZTYtYmNhYy00MTVjLTg3ZTQtMGJhMzRkNmNiNTBiIn0=",
        project_name="onopkosonja/byol",
        experiment_name="resnet50_not_pretrained")

    trainer = pl.Trainer(logger=neptune_logger, gpus=NUM_GPUS, max_epochs=EPOCHS, checkpoint_callback=checkpoint_callback)
    trainer.fit(model, TRAIN_LOADER, VAL_LOADER)
