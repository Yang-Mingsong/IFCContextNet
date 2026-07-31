import wandb

from utils.metrics import evaluate as run_eval


class Validator(object):
    def __init__(self, model, val_loader, loss_fn):
        self.model = model
        self.val_loader = val_loader
        self.loss_fn = loss_fn

    def evaluate(self, epoch, device, use_wandb=True):
        m = run_eval(self.model, self.val_loader, self.loss_fn, device)
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'val/loss': m['loss'], 'val/overall_acc': m['oa'], 'val/balanced_acc': m['ba'],
                'val/precision': m['pr'], 'val/recall': m['re'], 'val/f1': m['f1'],
            })
        print("Epoch{} - Val   Loss:{:.6f}, OA:{:.4f}, BA:{:.4f}, PR:{:.4f}, RE:{:.4f}, F1:{:.4f}".format(
            epoch, m['loss'], m['oa'], m['ba'], m['pr'], m['re'], m['f1']))
        return m