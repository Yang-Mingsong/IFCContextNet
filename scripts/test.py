import wandb

from utils.metrics import evaluate as run_eval


class Tester(object):
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn

    def evaluate(self, epoch, device, use_wandb=True, return_preds=False):
        m = run_eval(self.model, self.test_loader, self.loss_fn, device, return_preds=return_preds)
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'test/loss': m['loss'], 'test/overall_acc': m['oa'], 'test/balanced_acc': m['ba'],
                'test/precision': m['pr'], 'test/recall': m['re'], 'test/f1': m['f1'],
            })
        print("Epoch{} - Test  Loss:{:.6f}, OA:{:.4f}, BA:{:.4f}, PR:{:.4f}, RE:{:.4f}, F1:{:.4f}".format(
            epoch, m['loss'], m['oa'], m['ba'], m['pr'], m['re'], m['f1']))
        return m