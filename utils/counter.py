import argparse, os, glob, csv
from collections import Counter, defaultdict

def read_csv_rows(path):
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            with open(path, encoding=enc, newline='') as f:
                return [r for r in csv.reader(f) if r]
        except UnicodeDecodeError:
            continue
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'Q:\context5000\train')
    ap.add_argument('--cls', default='IfcCovering')
    ap.add_argument('--sample', type=int, default=3, help='打印几个非空样例的完整拆解')
    a = ap.parse_args()

    comp_dirs = sorted(glob.glob(os.path.join(a.root, a.cls, '*')))
    comp_dirs = [d for d in comp_dirs if os.path.isdir(d)]
    print(f'类别 {a.cls}: 发现 {len(comp_dirs)} 个构件目录')

    rel_counter = Counter()          # 关系实体类型 -> 出现构件数(按构件去重)
    rel_inst_counter = Counter()     # 关系实体类型 -> 实例总数
    slot_counter = Counter()         # 目标所在槽位(RelatedObjects/RelatingObject等) -> 次数
    partner_counter = Counter()      # 经关系相连的实体类别 -> 实例总数
    combo_counter = Counter()        # 每构件的"关系类型组合"指纹 -> 构件数
    n_empty = 0; n_missing = 0; n_ok = 0
    samples_printed = 0
    rows_out = []

    for i, d in enumerate(comp_dirs):
        if (i + 1) % 500 == 0:
            print(f'  ...已处理 {i+1}/{len(comp_dirs)}')
        npath = os.path.join(d, 'CONTEXT', 'semantics_node.csv')
        epath = os.path.join(d, 'CONTEXT', 'semantics_edge.csv')
        if not os.path.exists(npath):
            n_missing += 1
            rows_out.append([os.path.basename(d), 'MISSING', '', '', ''])
            continue
        nrows = read_csv_rows(npath)
        # 节点: kind,id,label
        cls_nodes = {}
        attr_nodes = {}
        for r in nrows:
            if len(r) < 3:
                continue
            kind, nid, label = r[0].strip(), r[1].strip(), r[2].strip()
            if kind == 'class_node':
                cls_nodes[nid] = label
            elif kind == 'attribute_node':
                attr_nodes[nid] = label
        # 空图: 仅目标一个class_node(或无节点)
        non_target_cls = {k: v for k, v in cls_nodes.items() if k != '0'}
        if len(cls_nodes) <= 1:
            n_empty += 1
            rows_out.append([os.path.basename(d), 'EMPTY', '', '', ''])
            continue
        n_ok += 1
        rels = sorted(v for v in non_target_cls.values() if v.startswith('IfcRel'))
        partners = sorted(v for v in non_target_cls.values()
                          if not v.startswith('IfcRel'))
        for rtype in set(rels):
            rel_counter[rtype] += 1
        rel_inst_counter.update(rels)
        partner_counter.update(partners)
        combo_counter['+'.join(sorted(set(rels))) if rels else '(无关系实体)'] += 1
        slots = []
        if os.path.exists(epath):
            for r in read_csv_rows(epath):
                if len(r) >= 5 and r[2].strip() == 'class_node' and r[3].strip() == '0' \
                        and r[4].strip().lower() == 'hasvalue' and r[0].strip() == 'attribute_node':
                    lbl = attr_nodes.get(r[1].strip(), '?')
                    slots.append(lbl)
        slot_counter.update(slots)
        rows_out.append([os.path.basename(d), 'OK',
                         '|'.join(sorted(set(rels))),
                         '|'.join(f'{k}x{v}' for k, v in Counter(partners).items()),
                         '|'.join(slots)])
        if samples_printed < a.sample:
            samples_printed += 1
            print(f'\n--- 样例 {samples_printed}: {os.path.basename(d)} ---')
            print(f'  关系实体: {Counter(rels)}')
            print(f'  相连实体: {Counter(partners)}')
            print(f'  目标槽位: {slots}')

    total = len(comp_dirs)
    print('\n================ 汇总 ================')
    print(f'构件总数 {total} | 非空 {n_ok} | 空图 {n_empty} ({100*n_empty/max(total,1):.2f}%) | 缺文件 {n_missing}')
    print('\n[关系实体类型] (出现的构件数 / 实例总数):')
    for k, v in rel_counter.most_common():
        print(f'  {k:36s} 构件 {v:5d} ({100*v/max(n_ok,1):.1f}% 非空构件) | 实例 {rel_inst_counter[k]}')
    print('\n[每构件关系组合指纹] 前8:')
    for k, v in combo_counter.most_common(8):
        print(f'  {k:60s} {v}')
    print('\n[目标所在槽位]:')
    for k, v in slot_counter.most_common():
        print(f'  {k:24s} {v}')
    print('\n[经关系相连的实体类别] 前15:')
    for k, v in partner_counter.most_common(15):
        print(f'  {k:36s} {v}')

    out = f'{a.cls}_sem_audit.csv'
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['component', 'status', 'relation_types', 'partner_classes', 'target_slots'])
        w.writerows(rows_out)

if __name__ == '__main__':
    main()