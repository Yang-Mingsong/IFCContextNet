import json
import os
import time

import torch


# ---------
# 参数量统计
# ---------
def count_parameters(model):
    def _n(m):
        return sum(p.numel() for p in m.parameters())

    stats = {'total': _n(model),
             'total_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}

    encoders = getattr(model, 'encoders', None)
    if encoders is not None:
        for name, enc in encoders.items():
            stats['encoder_' + name] = _n(enc)

    fusion = getattr(model, 'fusion', None)
    if fusion is not None:
        stats['fusion'] = _n(fusion)
    return stats


@torch.no_grad()
def benchmark_inference(model, loader, device, warmup_batches=5, move_batch_fn=None):
    if move_batch_fn is None:
        from utils.metrics import move_batch as move_batch_fn

    was_training = model.training
    model.eval()
    is_cuda = (device.type == 'cuda')

    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = move_batch_fn(batch, device)
        model(batch)
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    n_samples = 0
    forward_s = 0.0
    batch_forward_ms = []

    t_all0 = time.perf_counter()
    for batch in loader:
        batch = move_batch_fn(batch, device)
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(batch)
        if is_cuda:
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        forward_s += dt
        batch_forward_ms.append(dt * 1000.0)
        n_samples += int(batch['label'].shape[0])
    end2end_s = time.perf_counter() - t_all0

    res = {
        'num_samples': n_samples,
        'batch_size': getattr(loader, 'batch_size', None),
        'end2end_s': round(end2end_s, 4),
        'forward_s': round(forward_s, 4),
    }
    if n_samples > 0 and end2end_s > 0:
        res['throughput_end2end_elem_per_s'] = round(n_samples / end2end_s, 2)
        res['latency_end2end_ms_per_elem'] = round(end2end_s * 1000.0 / n_samples, 4)
    if n_samples > 0 and forward_s > 0:
        res['throughput_forward_elem_per_s'] = round(n_samples / forward_s, 2)
        res['latency_forward_ms_per_elem'] = round(forward_s * 1000.0 / n_samples, 4)

    if batch_forward_ms:
        srt = sorted(batch_forward_ms)
        res['batch_forward_ms_p50'] = round(srt[len(srt) // 2], 3)
        p95_idx = min(len(srt) - 1, int(round(0.95 * (len(srt) - 1))))
        res['batch_forward_ms_p95'] = round(srt[p95_idx], 3)

    if is_cuda:
        res['peak_mem_mb'] = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1)

    if was_training:
        model.train()
    return res


def _flatten(d, prefix='cost'):
    out = {}
    for k, v in d.items():
        key = '{}/{}'.format(prefix, k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def write_cost_summary(cost, out_dir, use_wandb=False):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'cost_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cost, f, ensure_ascii=False, indent=2)
    if use_wandb:
        import wandb
        wandb.log(_flatten(cost))
    print('[cost] summary written to {}'.format(path))
    return path
