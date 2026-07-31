import numpy as np
import torch
import sklearn.metrics as metrics


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if k == 'flag':
            out[k] = v
        else:
            out[k] = v.to(device)
    return out


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, return_preds=False):
    model.eval()
    total_loss, count = 0.0, 0
    y_true, y_pred = [], []
    flags = []
    logits_all = [] if return_preds else None

    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch['label']
        outs = model(batch)
        loss = loss_fn(outs, labels)
        pred = outs.argmax(dim=1)

        bs = labels.shape[0]
        total_loss += loss.item() * bs
        count += bs
        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        if return_preds:
            flags.extend(batch.get('flag', [''] * bs))
            logits_all.append(outs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    m = {
        'loss': total_loss / max(count, 1),
        'oa': metrics.accuracy_score(y_true, y_pred),
        'ba': metrics.balanced_accuracy_score(y_true, y_pred),
        'pr': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
        're': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': metrics.f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if return_preds:
        m['_preds'] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'logits': np.concatenate(logits_all) if logits_all else None,
            'flags': flags,
        }
    return m