import os
import sys
import csv

import numpy as np
import torch
import sklearn.metrics as M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if not hasattr(np, '_core'):
    import importlib
    for _name in ['multiarray', 'numeric', 'umath', '_multiarray_umath', 'numerictypes']:
        try:
            _m = importlib.import_module('numpy.core.' + _name)
            sys.modules.setdefault('numpy._core.' + _name, _m)
        except Exception:
            pass
    try:
        sys.modules.setdefault('numpy._core', importlib.import_module('numpy.core'))
    except Exception:
        pass


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _heatmap(mat, classes, title, out_png, normalize=False, fmt=None):
    data = mat.astype(float)
    if normalize:
        row = data.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        data = data / row
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.45)))
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    # 类别不太多时把数值标上
    if n <= 25:
        f = fmt or ('{:.2f}' if normalize else '{:.0f}')
        thr = data.max() * 0.6
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f.format(data[i, j]), ha='center', va='center',
                        fontsize=6, color='white' if data[i, j] < thr else 'black')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def analyze(run_dir):
    d = torch.load(os.path.join(run_dir, 'test_preds.pt'), weights_only=False)
    y_true = _to_np(d['y_true'])
    y_pred = _to_np(d['y_pred'])
    classes = list(d['classes'])
    flags = d.get('flags', None)
    labels_idx = list(range(len(classes)))

    # 混淆矩阵
    cm = M.confusion_matrix(y_true, y_pred, labels=labels_idx)
    _heatmap(cm, classes, 'Confusion (row-normalized)',
             os.path.join(run_dir, 'confusion_matrix_norm.png'), normalize=True)
    _heatmap(cm, classes, 'Confusion (raw counts)',
             os.path.join(run_dir, 'confusion_matrix_raw.png'), normalize=False)

    # 细类报告
    rep = M.classification_report(y_true, y_pred, labels=labels_idx,
                                  target_names=classes, zero_division=0, output_dict=True)
    rows = [(c, rep[c]['precision'], rep[c]['recall'], rep[c]['f1-score'], int(rep[c]['support']))
            for c in classes]
    rows.sort(key=lambda r: r[3])  # 按 f1 升序
    with open(os.path.join(run_dir, 'per_class_report.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        for c, p, r, f1, s in rows:
            w.writerow([c, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", s])

    # 最易混淆类对
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                denom = cm[i].sum()
                pairs.append((classes[i], classes[j], int(cm[i, j]),
                              cm[i, j] / denom if denom else 0.0))
    pairs.sort(key=lambda x: x[2], reverse=True)
    with open(os.path.join(run_dir, 'top_confusions.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['true', 'pred_as', 'count', 'frac_of_true_class'])
        for t, p, c, fr in pairs[:50]:
            w.writerow([t, p, c, f"{fr:.4f}"])

    oa = M.accuracy_score(y_true, y_pred)
    ba = M.balanced_accuracy_score(y_true, y_pred)
    f1m = M.f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"[{os.path.basename(run_dir)}] OA={oa:.4f} BA={ba:.4f} macroF1={f1m:.4f}  N={len(y_true)}")
    print("最弱 5 类(按 f1)：")
    for c, p, r, f1, s in rows[:5]:
        print(f"  {c:<24} f1={f1:.3f} recall={r:.3f} precision={p:.3f} support={s}")
    print("最易混淆 5 对(真->被预测成)：")
    for t, p, c, fr in pairs[:5]:
        print(f"  {t:<20} -> {p:<20} {c} 次 ({fr:.1%})")
    print(f"图/表已写入：{run_dir}")

if __name__ == '__main__':
    analyze(r"Q:\pychem_project\IFCContextNet_Master\results\test1000_rep+sem+spa_hop2_REVERSE")
