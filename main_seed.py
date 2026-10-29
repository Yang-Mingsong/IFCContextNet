import os
import csv
import random
import numpy as np
import torch
import dgl
import sklearn.metrics as M
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.custom_dataset import BIMContextNetDataset, collate_modalities
from models.model import IFCComponentClassifier
from scripts.train import Trainer
from scripts.test import Tester


def set_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except Exception:
        pass

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    print(f"[seed] Random seed = {seed}")


def _save_artifacts(cfg, model, classes, preds):
    out_dir = os.path.join(cfg.get('results_dir', 'result'), cfg['run_name'])
    os.makedirs(out_dir, exist_ok=True)

    torch.save({
        'state_dict': model.state_dict(),
        'classes': classes,
        'modalities': cfg['modalities'],
        'n_hops': cfg['n_hops'],
        'edge_direction': cfg['edge_direction'],
        'epoch': cfg['num_epochs'],
        'seed': int(cfg['seed']),
        'run_name': cfg['run_name'],
    }, os.path.join(out_dir, f"model_epoch{cfg['num_epochs']}.pt"))

    torch.save({
        'y_true': preds['y_true'],
        'y_pred': preds['y_pred'],
        'logits': preds['logits'],
        'flags': preds['flags'],
        'classes': classes,
        'seed': int(cfg['seed']),
    }, os.path.join(out_dir, 'test_preds.pt'))

    labels_idx = list(range(len(classes)))
    cm = M.confusion_matrix(preds['y_true'], preds['y_pred'], labels=labels_idx)
    with open(os.path.join(out_dir, 'confusion_matrix.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['true\\pred'] + list(classes))
        for i, row in enumerate(cm):
            w.writerow([classes[i]] + row.tolist())

    rep = M.classification_report(
        preds['y_true'], preds['y_pred'],
        labels=labels_idx,
        target_names=list(classes),
        zero_division=0,
        output_dict=True,
    )
    with open(os.path.join(out_dir, 'per_class_report.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for c in classes:
            r = rep[c]
            w.writerow([
                c,
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                int(r['support']),
            ])

    print(
        f"[saved] {out_dir}: model_epoch{cfg['num_epochs']}.pt, "
        "test_preds.pt, confusion_matrix.csv, per_class_report.csv"
    )


def run_one(cfg):
    set_seed(cfg['seed'])

    use_wandb = cfg.get('use_wandb', True)
    if use_wandb:
        wandb.init(
            project=cfg['project_name'],
            name=cfg['run_name'],
            config=cfg,
            reinit=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(False)

    print('Loading datasets...')
    train_dataset = BIMContextNetDataset(
        root=cfg['train_data_dir'],
        split='train',
        cache_dir=cfg['cache_dir'],
        isg=cfg['isg_name'],
        edge_direction=cfg['edge_direction'],
        modalities=cfg['modalities'],
        n_hops=cfg['n_hops'],
        build_max_hop=cfg['build_max_hop'],
    )
    test_dataset = BIMContextNetDataset(
        root=cfg['test_data_dir'],
        split='test',
        cache_dir=cfg['cache_dir'],
        isg=cfg['isg_name'],
        edge_direction=cfg['edge_direction'],
        modalities=cfg['modalities'],
        n_hops=cfg['n_hops'],
        build_max_hop=cfg['build_max_hop'],
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(int(cfg['seed']))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train_batch_size'],
        shuffle=True,
        drop_last=False,
        collate_fn=collate_modalities,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['test_batch_size'],
        shuffle=False,
        drop_last=False,
        collate_fn=collate_modalities,
    )

    classes = train_dataset.classes
    print('Creating model...')
    model = IFCComponentClassifier(
        modalities=cfg['modalities'],
        in_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        out_dim=cfg['output_dim'],
        num_classes=len(classes),
        sem_availability_only=cfg.get('sem_availability_only', False),
        rep_metagraph=train_dataset.rep_metagraph,
        sem_metagraph=train_dataset.sem_metagraph_,
        n_hops=cfg['n_hops'],
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
        betas=(0.9, 0.999),
    )
    loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(
        optimizer,
        cfg['num_epochs'],
        eta_min=cfg['eta_min'],
    )

    trainer = Trainer(model, train_loader, optimizer, loss)
    tester = Tester(model, test_loader, loss)

    print(f"start training. run = {cfg['run_name']}")
    for epoch in range(1, cfg['num_epochs'] + 1):
        trainer.train(epoch, device, use_wandb)
        scheduler.step()

    result = tester.evaluate(
        cfg['num_epochs'],
        device,
        use_wandb,
        return_preds=True,
    )
    result['epoch'] = cfg['num_epochs']

    preds = result.pop('_preds', None)
    if preds is not None:
        _save_artifacts(cfg, model, classes, preds)

    if use_wandb:
        wandb.finish()

    torch.cuda.empty_cache()
    return result


def bim_context_main(
    config_path=r'Q:\pychem_project\IFCContextNet_Master\configs\configs_seed.yaml'
):
    from utils.grid_seed import run_grid
    run_grid(config_path, run_one_fn=run_one)


if __name__ == '__main__':
    bim_context_main()
