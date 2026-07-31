import os
import csv
import itertools

import yaml


def _expand(grid):
    """笛卡尔积展开：datasets × modalities × n_hops × edge_direction"""
    for ds, mod, hop, ed in itertools.product(
            grid['datasets'], grid['modalities'], grid['n_hops'], grid['edge_direction']):
        yield ds, mod, hop, ed


def run_grid(config_path):
    # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！
    # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！
    # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！
    from main_null import run_one   # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！
    # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！
    # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！
    # 延迟导入，避免循环依赖  ！！！！！！更换main时记着改这个地方！！！！！！

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    base = dict(config['ContextNet'])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(base['gpu_device'])

    grid = base['grid']
    ds_paths = base['datasets']

    os.makedirs(base['results_dir'], exist_ok=True)
    out_csv = os.path.join(base['results_dir'], 'grid_results.csv')

    # 剩下的是每个 run 共用的扁平超参
    flat = {k: v for k, v in base.items() if k not in ('grid', 'datasets')}

    rows = []
    for ds, mod, hop, ed in _expand(grid):
        paths = ds_paths[ds]
        run_name = "{}_{}_hop{}_{}".format(ds, '+'.join(mod), hop, ed)

        cfg = dict(flat)
        cfg.update({
            'train_data_dir': paths['train_data_dir'],
            'test_data_dir': paths['test_data_dir'],
            'cache_dir': paths['cache_dir'],
            'modalities': list(mod),
            'n_hops': hop,
            'edge_direction': ed,
            'run_name': run_name,
        })

        print("\n==== RUN: {} ====".format(run_name))
        m = run_one(cfg)

        rows.append({
            'dataset': ds,
            'modalities': '+'.join(mod),
            'n_hops': hop,
            'edge_dir': ed,
            'epoch': m.get('epoch'),
            'f1': round(m['f1'], 6),
            'ba': round(m['ba'], 6),
            'pr': round(m['pr'], 6),
            're': round(m['re'], 6),
            'oa': round(m['oa'], 6),
            'loss': round(m['loss'], 6),
        })

        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nGrid done ({} runs). CSV -> {}".format(len(rows), out_csv))
    return out_csv