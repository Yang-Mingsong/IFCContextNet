import numpy as np
import torch
import wandb
import sklearn.metrics as metrics
from tqdm import tqdm

from utils.metrics import move_batch


class Trainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, epoch, device, use_wandb=True):
        self.model.train()
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} train'.format(epoch))

        total_loss, count = 0.0, 0
        y_true, y_pred = [], []

        for batch in tqdm_batch:
            batch = move_batch(batch, device)
            labels = batch['label']

            self.optimizer.zero_grad(set_to_none=True)
            outs = self.model(batch)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().argmax(dim=1)
            bs = labels.shape[0]
            total_loss += loss.item() * bs
            count += bs
            y_true.append(labels.cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        m = {
            'loss': total_loss / max(count, 1),
            'oa': metrics.accuracy_score(y_true, y_pred),
            'ba': metrics.balanced_accuracy_score(y_true, y_pred),
        }

        if use_wandb:
            wandb.log({'epoch': epoch, 'train/loss': m['loss'],
                       'train/overall_acc': m['oa'], 'train/balanced_acc': m['ba']})
        print("Epoch{} - Train Loss:{:.6f}, OA:{:.4f}, BA:{:.4f}".format(
            epoch, m['loss'], m['oa'], m['ba']))

        torch.cuda.empty_cache()
        return m
