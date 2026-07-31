import os
import random
import re
from collections import defaultdict
import torch
import pandas as pd

STEM_RE = re.compile(r'^(?P<building_type>\d+)_(?P<cls>[A-Za-z]+)_(?P<idx>\d+)$')


def _load_official_csv(path, split):
    df = pd.read_csv(path, encoding='utf-8-sig')
    stem = df['Train_File'].astype(str).str.replace(r'\.obj$', '', regex=True)
    parsed = stem.str.extract(STEM_RE)
    if parsed['cls'].isna().any():
        bad = df.loc[parsed['cls'].isna(), 'Train_File'].head(5).tolist()
        raise ValueError(f'Train_File 无法解析，例如: {bad}')

    def proj_key(p):
        parts = str(p).replace('/', '\\').split('\\')
        if 'ALL_IFC_MODEL' in parts:
            i = parts.index('ALL_IFC_MODEL')
            return parts[i + 1] + '/' + parts[i + 2]
        return '\\'.join(parts[:-1])   # 兜底：取父目录

    return pd.DataFrame({
        'flag': split + stem,
        'cls': parsed['cls'],
        'building_type': parsed['building_type'],
        'project': df['Mapped_IFC_Path'].apply(proj_key),
        'ifc_file': df['Mapped_IFC_Path'],
        'orig_split': split,
    })


def _join_with_cache(df, cache_dir):
    cache_flags = []
    for split in ('train', 'test'):
        path = os.path.join(cache_dir, split, 'fusion_flags.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'找不到 {path}')
        cache_flags += [str(f) for f in torch.load(path, weights_only=True)]
    cache_set = set(cache_flags)
    csv_set = set(df['flag'])

    only_cache = sorted(cache_set - csv_set)
    only_csv = len(csv_set - cache_set)
    kept = df[df['flag'].isin(cache_set)].copy()

    print(f'[join] 缓存样本 {len(cache_set)}，CSV 行 {len(csv_set)}，匹配 {len(kept)}')
    if only_csv:
        print(f'[join] CSV 独有 {only_csv} 行')
    if only_cache:
        print(f'[join][WARNING] 缓存独有 {len(only_cache)} 个样本在 CSV 中无项目映射，例如: {only_cache[:5]}')
        raise RuntimeError('存在无项目映射的缓存样本')
    return kept


def _search_split(df, test_frac, seed, trials, min_class_test, min_class_train):
    total = len(df)
    target_n = round(total * test_frac)

    proj_size = df.groupby('project').size().to_dict()
    projects = sorted(proj_size)
    cls_total = df.groupby('cls').size().to_dict()
    pc = defaultdict(lambda: defaultdict(int))
    for _, r in df.iterrows():
        pc[r['project']][r['cls']] += 1

    rng = random.Random(seed)
    best = None
    fail_count = defaultdict(int)

    for _ in range(trials):
        order = projects[:]
        rng.shuffle(order)
        test_projects, n = [], 0
        for p in order:
            if n >= target_n:
                break
            test_projects.append(p)
            n += proj_size[p]

        cls_test = defaultdict(int)
        for p in test_projects:
            for c, k in pc[p].items():
                cls_test[c] += k

        feasible = True
        for c, tot in cls_total.items():
            te = cls_test.get(c, 0)
            tr = tot - te
            if te < min_class_test or tr < min_class_train:
                fail_count[c] += 1
                feasible = False
        if not feasible:
            continue

        score = sum((cls_test.get(c, 0) / tot - test_frac) ** 2 for c, tot in cls_total.items())
        score += abs(n / total - test_frac)
        if best is None or score < best[0]:
            best = (score, sorted(test_projects), n)

    if best is None:
        top = sorted(fail_count.items(), key=lambda x: -x[1])[:8]
        raise RuntimeError(f'{trials} 次尝试内未找到可行解。最常违反约束的类：{top}\n')
    return best


def make_project_split(csv_train, csv_test,
                       cache_dir='',
                       out_dir='',
                       test_frac=0.2,
                       seed=42,
                       trials=5000,
                       min_class_test=20,
                       min_class_train=100,
                       exclude_classes=()):
    df = pd.concat([_load_official_csv(csv_train, 'train'),
                    _load_official_csv(csv_test, 'test')], ignore_index=True)

    if cache_dir:
        df = _join_with_cache(df, cache_dir)
    else:
        print('[warn] 未提供 cache_dir')

    exclude_classes = list(exclude_classes)
    if exclude_classes:
        before = len(df)
        df = df[~df['cls'].isin(exclude_classes)].copy()
        print(f'[info] 显式排除 {exclude_classes}，样本 {before} -> {len(df)}')

    proj_per_cls = df.groupby('cls')['project'].nunique()
    stuck = proj_per_cls[proj_per_cls < 2]

    print(f'[info] 样本 {len(df)}，类 {df["cls"].nunique()}，项目 {df["project"].nunique()}，'
          f'目标测试比例 {test_frac:.0%}')

    score, test_projects, n_test = _search_split(df, test_frac, seed, trials,
                                                 min_class_test, min_class_train)

    is_test = df['project'].isin(set(test_projects))
    new_test, new_train = df[is_test], df[~is_test]

    print(f'\n[result] held-out 项目 {len(test_projects)} 个，新 test {len(new_test)}'
          f'（{len(new_test)/len(df):.1%}），新 train {len(new_train)}，score={score:.5f}')
    print(f'\n{"class":<26s}{"train":>8s}{"test":>8s}{"test%":>8s}')
    for c, g in df.groupby('cls'):
        te = int(is_test[g.index].sum())
        tr = len(g) - te
        print(f'{c:<26s}{tr:>8d}{te:>8d}{te/len(g):>8.1%}')

    if not out_dir:
        print('\n[info] 未指定 out_dir')
        return None

    os.makedirs(out_dir, exist_ok=True)
    cols = ['flag', 'cls', 'building_type', 'project', 'orig_split']
    for name, subset in (('manifest_train.csv', new_train), ('manifest_test.csv', new_test)):
        path = os.path.join(out_dir, name)
        subset.sort_values('flag')[cols].rename(columns={'cls': 'class'}).to_csv(
            path, index=False, encoding='utf-8-sig')
        print(f'[saved] {path} ({len(subset)} rows)')

    with open(os.path.join(out_dir, 'split_protocol.txt'), 'w', encoding='utf-8') as f:
        f.write('protocol: project-held-out split\n')
        f.write('grouping unit: project folder (two levels under ALL_IFC_MODEL)\n')
        f.write(f'seed={seed}, trials={trials}, target test fraction={test_frac}\n')
        f.write(f'hard constraints: per-class test>={min_class_test}, train>={min_class_train}\n')
        f.write('objective: minimize sum of squared per-class deviation from target fraction\n')
        f.write(f'excluded classes: {exclude_classes or "none"}\n')
        f.write(f'result: {len(test_projects)} held-out projects, test={len(new_test)} '
                f'({len(new_test)/len(df):.3f}), train={len(new_train)}\n\n')
        f.write('held-out projects:\n')
        for p in test_projects:
            f.write(f'  {p}\n')
    print(f"[saved] {os.path.join(out_dir, 'split_protocol.txt')}")
    return out_dir


if __name__ == '__main__':
    make_project_split(
        csv_train=r'Q:\pychem_project\IFCContextNet_Master\data\BIMCompNet_1000_train.csv',
        csv_test=r'Q:\pychem_project\IFCContextNet_Master\data\BIMCompNet_1000_test.csv',
        cache_dir=r'Q:\cache1000',
        out_dir=r'Q:\splits\project_heldout_1000_v1',
        test_frac=0.2,
        seed=42,
        trials=5000,
        min_class_test=20,
        min_class_train=100,
        exclude_classes=(),
    )