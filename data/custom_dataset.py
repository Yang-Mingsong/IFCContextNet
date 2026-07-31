import os
import re
import torch
import concurrent.futures
import dgl
import pandas as pd
from dgl.data import DGLDataset
from collections import defaultdict
from dgl import save_graphs, load_graphs


MODALITY_BIN = {
    'rep': 'rep_graphs.bin',
    'spa': 'spa_graphs.bin',
    'sem': 'sem_graphs.bin',
}
VALID_MODALITIES = ('rep', 'sem', 'spa')


def find_classes(_dir):
    classes = [d for d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class BIMContextNetDataset(DGLDataset):
    def __init__(self, root, split, cache_dir, isg, edge_direction,
                 modalities=('rep', 'sem', 'spa'), n_hops=2, build_max_hop=5):
        self.root = root
        self.classes, self.class_to_idx = find_classes(root)
        self.split = split
        self._cache_path = cache_dir
        self.isg = isg
        self.edge_direction = edge_direction

        # 模态选择
        modalities = [m for m in VALID_MODALITIES if m in set(modalities)]
        self.modalities = modalities

        # spa 读取跳数
        self.n_hops = int(n_hops)
        # spa 缓存的最大跳数
        self.build_max_hop = int(build_max_hop)

        self.rep_graphs = []
        self.spa_graphs = []
        self.sem_graphs = []
        self.labels = []
        self.model_flag = []

        self.metagraph = [
            ('class_node', 'hasAttribute', 'attribute_node'),
            ('attribute_node', 'hasValue', 'class_node'),
            ('attribute_node', 'hasValue', 'type_node'),
            ('attribute_node', 'hasValue', 'value_node'),
            ('type_node', 'hasValue', 'class_node'),
            ('type_node', 'hasValue', 'type_node'),
            ('type_node', 'hasValue', 'value_node'),
            ('class_node', 'selfLoop', 'class_node'),
            ('attribute_node', 'selfLoop', 'attribute_node'),
            ('type_node', 'selfLoop', 'type_node'),
            ('value_node', 'selfLoop', 'value_node')
        ]
        self.re_metagraph = [
            ('attribute_node', 'hasAttribute', 'class_node'),
            ('class_node', 'hasValue', 'attribute_node'),
            ('type_node', 'hasValue', 'attribute_node'),
            ('value_node', 'hasValue', 'attribute_node'),
            ('class_node', 'hasValue', 'type_node'),
            ('type_node', 'hasValue', 'type_node'),
            ('value_node', 'hasValue', 'type_node'),
            ('class_node', 'selfLoop', 'class_node'),
            ('attribute_node', 'selfLoop', 'attribute_node'),
            ('type_node', 'selfLoop', 'type_node'),
            ('value_node', 'selfLoop', 'value_node')
        ]
        self.re_sem_metagraph = [
            ('attribute_node', 'hasAttribute', 'class_node'),
            ('class_node', 'hasValue', 'attribute_node'),
            ('type_node', 'hasValue', 'attribute_node'),
            ('class_node', 'hasValue', 'type_node'),
            ('type_node', 'hasValue', 'type_node'),
            ('class_node', 'inverseAttribute', 'class_node'),
            ('class_node', 'selfLoop', 'class_node'),
            ('attribute_node', 'selfLoop', 'attribute_node'),
            ('type_node', 'selfLoop', 'type_node'),
        ]
        self.sem_metagraph = [
            ('class_node', 'hasAttribute', 'attribute_node'),
            ('attribute_node', 'hasValue', 'class_node'),
            ('attribute_node', 'hasValue', 'type_node'),
            ('type_node', 'hasValue', 'class_node'),
            ('type_node', 'hasValue', 'type_node'),
            ('class_node', 'inverseAttribute', 'class_node'),
            ('class_node', 'selfLoop', 'class_node'),
            ('attribute_node', 'selfLoop', 'attribute_node'),
            ('type_node', 'selfLoop', 'type_node'),
        ]

        super(BIMContextNetDataset, self).__init__(root)

    @property
    def rep_metagraph(self):
        return self.re_metagraph if self.edge_direction == 'REVERSE' else self.metagraph

    @property
    def sem_metagraph_(self):
        return self.re_sem_metagraph if self.edge_direction == 'REVERSE' else self.sem_metagraph

    # -------------------------
    # utils (rep / value_node)
    # -------------------------
    @staticmethod
    def _is_number(obj: str) -> bool:
        pattern = r'^[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?$'
        return bool(re.match(pattern, obj))

    @staticmethod
    def _is_integer(obj: str) -> bool:
        pattern = r'^[-+]?[0-9]+$'
        return bool(re.match(pattern, obj))

    @staticmethod
    def _is_bool(obj: str) -> bool:
        return obj in {'True', 'False'}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + torch.exp(-torch.tensor(x)))

    # ===========
    # 多进程构建
    # ===========
    def _run_multiprocess(self, path_list, worker_fn, tag=""):
        results = {}
        label_to_list = {}
        for label, p in path_list:
            label_to_list.setdefault(label, []).append((label, p))

        num_processes = 24
        for label, sub_list in label_to_list.items():
            graph_path_list = sub_list
            while True:
                results_temp = {}
                failed_files = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                    future_to_file = {executor.submit(worker_fn, file): file for file in graph_path_list}
                    for future in concurrent.futures.as_completed(future_to_file):
                        try:
                            res = future.result()
                            results_temp[label] = results_temp.get(label, []) + [res]
                        except Exception as exc:
                            file = future_to_file[future]
                            print(f'[{tag}] {file},{exc}')
                            failed_files.append(file)

                if label in results and results[label]:
                    results[label] += results_temp.get(label, [])
                else:
                    results[label] = results_temp.get(label, [])

                if not failed_files:
                    print(f"[{tag}] {len(results.get(label, []))} 更新完了 {len(graph_path_list)}")
                    break

                graph_path_list = [(label, file) for _, file in failed_files]

        return results

    @staticmethod
    def _flatten_results(results):
        """results(dict[label]->list[list[(flag, g, y)]]) -> dict[flag] -> (g, y)"""
        m = {}
        for _, lst in results.items():
            for graphs in lst:
                for (mflag, g, y) in graphs:
                    m[mflag] = (g, y)
        return m

    # --------------
    # rep builders
    # --------------
    def _process_rep_label_path(self, label_path):
        label, _path = label_path
        graphs = []
        a = ''
        for root, dirs, files in os.walk(_path):
            node_file_path = ''
            edge_file_path = ''
            bin_file_path = ''
            for file in files:
                if file.endswith('geo_node.csv'):
                    node_file_path = os.path.join(root, file)
                elif file.endswith('geo_edge.csv'):
                    edge_file_path = os.path.join(root, file)
                elif file.endswith(self.isg):
                    bin_file_path = os.path.join(root, file)

            if node_file_path != '' and edge_file_path != '':
                parts = root.split('\\')
                _model_flag = parts[2] + parts[4]
                _shared_embedding_information = torch.load(str(bin_file_path), weights_only=False)
                gra = self._load_hetero_graph_from_csv(
                    edge_file_path, node_file_path, label, _shared_embedding_information, _model_flag
                )
                graphs.append(gra)
                a = node_file_path
        print(' {}已完成'.format(a))
        return graphs

    def _load_hetero_graph_from_csv(self, edge_file_path, node_file_path, label, _shared_embedding_information, model_flag):
        all_edge_types = self.metagraph
        edge_dict = {etype: ([], []) for etype in all_edge_types}

        edges = pd.read_csv(edge_file_path, header=None)
        for _, row in edges.iterrows():
            src_type, src_id, dst_type, dst_id, edge_type = row
            edge_key = (str(src_type), str(edge_type), str(dst_type))
            edge_dict[edge_key][0].append(int(src_id))
            edge_dict[edge_key][1].append(int(dst_id))

        g = dgl.heterograph(edge_dict)

        nodes_infor = pd.read_csv(node_file_path, header=None)
        node_feat_dict = defaultdict(dict)
        for _, row in nodes_infor.iterrows():
            node_type, node_id, item = row
            if node_type == 'class_node':
                features = _shared_embedding_information['class_node'][item]
            elif node_type == 'attribute_node':
                features = _shared_embedding_information['attribute_node'][item]
            elif node_type == 'type_node':
                if item == 'TUPLE':
                    features = _shared_embedding_information['type_node']['LIST']
                elif item == 'TUPLETUPLE':
                    features = _shared_embedding_information['type_node']['LISTLIST']
                else:
                    features = _shared_embedding_information['type_node'][item]
            elif node_type == 'value_node':
                if item.upper() in _shared_embedding_information['value_node'].keys():
                    features = _shared_embedding_information['value_node'][item.upper()]
                elif self._is_integer(str(item)):
                    features = self._sigmoid(float(item)) * _shared_embedding_information['type_node']['INTEGER']
                elif self._is_number(str(item)):
                    features = self._sigmoid(float(item)) * _shared_embedding_information['type_node']['REAL']
                elif self._is_bool(str(item)):
                    scaled_value = 1.0 if item == 'True' else 0.0
                    features = scaled_value * _shared_embedding_information['type_node']['BOOLEAN']
                else:
                    features = _shared_embedding_information['type_node']['STRING']
            else:
                print(node_type, node_id, item)
                features = _shared_embedding_information[node_type][item]
            node_feat_dict[node_type][node_id] = features

        for ntype in node_feat_dict:
            node_ids = list(node_feat_dict[ntype].keys())
            features_list = [node_feat_dict[ntype][nid] for nid in node_ids]
            feats = torch.stack(features_list)
            if g.num_nodes(ntype) != len(feats):
                print("不相等：Node type: {}, Number of nodes: {}, Number of features: {}".format(
                    ntype, g.num_nodes(ntype), len(feats)))
            g.nodes[ntype].data['features'] = feats

        y = self.class_to_idx[label]
        return model_flag, g, y

    # ---------------
    # sem builders
    # ---------------
    def _process_sem_label_path(self, label_path):
        label, _path = label_path
        graphs = []
        a = ''
        for root, dirs, files in os.walk(_path):
            node_file_path = ''
            edge_file_path = ''
            bin_file_path = ''
            for file in files:
                if file.endswith('semantics_node.csv'):
                    node_file_path = os.path.join(root, file)
                elif file.endswith('semantics_edge.csv'):
                    edge_file_path = os.path.join(root, file)
                elif file.endswith(self.isg):
                    bin_file_path = os.path.join(root, file)

            if node_file_path != '' and edge_file_path != '':
                parts = root.split('\\')
                _model_flag = parts[2] + parts[4]
                _shared_embedding_information = torch.load(str(bin_file_path), weights_only=False)
                gra = self._load_sem_graph_from_csv(
                    edge_file_path, node_file_path, label, _shared_embedding_information, _model_flag
                )
                graphs.append(gra)
                a = node_file_path
        print(' {}已完成'.format(a))
        return graphs

    def _load_sem_graph_from_csv(self, edge_file_path, node_file_path, label, _shared_embedding_information, model_flag):
        all_edge_types = self.sem_metagraph
        edge_dict = {etype: ([], []) for etype in all_edge_types}

        if edge_file_path and os.path.exists(edge_file_path) and os.path.getsize(edge_file_path) > 0:
            edges = pd.read_csv(edge_file_path, header=None)
        else:
            edges = pd.DataFrame(columns=[0, 1, 2, 3, 4])
        for _, row in edges.iterrows():
            src_type, src_id, dst_type, dst_id, edge_type = row
            edge_key = (str(src_type), str(edge_type), str(dst_type))
            edge_dict[edge_key][0].append(int(src_id))
            edge_dict[edge_key][1].append(int(dst_id))

        # 反属性边：target(0) -> 每个 rel_node
        ia_key = ('class_node', 'inverseAttribute', 'class_node')
        fwd_attr_key = ('class_node', 'hasAttribute', 'attribute_node')
        rel_ids = set(edge_dict[fwd_attr_key][0])
        rel_ids.discard(0)
        for rid in rel_ids:
            edge_dict[ia_key][0].append(0)
            edge_dict[ia_key][1].append(rid)

        nodes_infor = pd.read_csv(node_file_path, header=None)
        type_to_ids = defaultdict(list)
        for _, row in nodes_infor.iterrows():
            node_type, node_id, _ = row
            type_to_ids[node_type].append(int(node_id))

        for ntype, ids in type_to_ids.items():
            etype = (ntype, 'selfLoop', ntype)
            for nid in ids:
                edge_dict[etype][0].append(nid)
                edge_dict[etype][1].append(nid)

        all_ntypes_in_meta = set()
        for src, _, dst in self.sem_metagraph:
            all_ntypes_in_meta.add(src)
            all_ntypes_in_meta.add(dst)
        num_nodes_dict = {nt: 0 for nt in all_ntypes_in_meta}
        for ntype, ids in type_to_ids.items():
            if ntype in num_nodes_dict:
                num_nodes_dict[ntype] = max(ids) + 1

        g = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

        for ntype, expected in num_nodes_dict.items():
            if ntype not in g.ntypes:
                continue
            actual = g.num_nodes(ntype)
            if actual < expected:
                g.add_nodes(expected - actual, ntype=ntype)

        nodes_infor = pd.read_csv(node_file_path, header=None)
        node_feat_dict = defaultdict(dict)
        for _, row in nodes_infor.iterrows():
            node_type, node_id, item = row
            if node_type == 'class_node':
                if node_id == 0:
                    features = _shared_embedding_information['class_node']['IfcElement']
                else:
                    features = _shared_embedding_information['class_node'][item]
            elif node_type == 'attribute_node':
                features = _shared_embedding_information['attribute_node'][item]
            elif node_type == 'type_node':
                if item == 'TUPLE':
                    features = _shared_embedding_information['type_node']['LIST']
                elif item == 'TUPLETUPLE':
                    features = _shared_embedding_information['type_node']['LISTLIST']
                else:
                    features = _shared_embedding_information['type_node'][item]
            else:
                print(node_type, node_id, item)
                features = _shared_embedding_information[node_type][item]
            node_feat_dict[node_type][node_id] = features

        feat_dim = None
        sample_feat = None
        for ntype, ids_dict in node_feat_dict.items():
            if len(ids_dict) > 0:
                sample_feat = next(iter(ids_dict.values()))
                feat_dim = sample_feat.shape[-1]
                break

        for ntype in g.ntypes:
            if ntype in node_feat_dict and len(node_feat_dict[ntype]) > 0:
                node_ids = list(node_feat_dict[ntype].keys())
                features_list = [node_feat_dict[ntype][nid] for nid in node_ids]
                feats = torch.stack(features_list)
            else:
                feats = torch.zeros(g.num_nodes(ntype), feat_dim, dtype=sample_feat.dtype)
            if g.num_nodes(ntype) != len(feats):
                print("不相等：Node type: {}, Number of nodes: {}, Number of features: {}".format(
                    ntype, g.num_nodes(ntype), len(feats)))
            g.nodes[ntype].data['features'] = feats

        y = self.class_to_idx[label]
        return model_flag, g, y

    # --------------
    # spa builders
    # --------------
    def _process_spa_label_path(self, label_path):
        label, _path = label_path
        graphs = []
        a = ''
        for root, dirs, files in os.walk(_path):
            node_file_path = ''
            edge_file_path = ''
            bin_file_path = ''
            for file in files:
                if file.endswith('geometry_node.csv'):
                    node_file_path = os.path.join(root, file)
                elif file.endswith('geometry_edge.csv'):
                    edge_file_path = os.path.join(root, file)
                elif file.endswith(self.isg):
                    bin_file_path = os.path.join(root, file)

            if node_file_path != '' and edge_file_path != '':
                parts = root.split('\\')
                _model_flag = parts[2] + parts[4]
                _shared_embedding_information = torch.load(str(bin_file_path), weights_only=False)
                gra = self._load_homo_graph_from_csv(
                    edge_file_path, node_file_path, label, _shared_embedding_information, _model_flag
                )
                graphs.append(gra)
                a = node_file_path
        print(' {}已完成'.format(a))
        return graphs

    def _load_homo_graph_from_csv(self, edge_file_path, node_file_path, label, _shared_embedding_information, model_flag):
        # 节点
        nodes_df = pd.read_csv(node_file_path, header=None).iloc[:, :3]
        nodes_df.columns = ['node_type', 'node_id', 'item']
        nodes_df['node_id'] = nodes_df['node_id'].astype(int)

        # 边
        if edge_file_path and os.path.exists(edge_file_path) and os.path.getsize(edge_file_path) > 0:
            edges_df = pd.read_csv(edge_file_path, header=None)
        else:
            edges_df = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5])
        edges_df = edges_df.iloc[:, :6]
        edges_df.columns = ['src_type', 'src_id', 'dst_type', 'dst_id', 'edge_type', 'hop']
        edges_df['src_id'] = edges_df['src_id'].astype(int)
        edges_df['dst_id'] = edges_df['dst_id'].astype(int)
        edges_df['hop'] = edges_df['hop'].astype(int)

        # 保留到 build_max_hop
        edges_df = edges_df[edges_df['hop'] <= self.build_max_hop].copy()
        edges_df = edges_df.sort_values('hop').drop_duplicates(subset=['src_id', 'dst_id'], keep='first')

        # 只保留 <=build_max_hop 内涉及的节点；target 节点必留
        selected_node_ids = set(edges_df['src_id'].tolist()) | set(edges_df['dst_id'].tolist())
        selected_node_ids.add(0)
        nodes_df = nodes_df[nodes_df['node_id'].isin(selected_node_ids)].copy()
        nodes_df = nodes_df.sort_values('node_id').reset_index(drop=True)
        if len(nodes_df) == 0:
            raise ValueError(f'geometry_node.csv 中没有可用节点: {node_file_path}')

        # node_id 重映射
        old_ids = nodes_df['node_id'].tolist()
        id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
        nodes_df['new_id'] = nodes_df['node_id'].map(id_map)
        num_nodes = len(nodes_df)

        # 节点特征
        embed_dim = next(iter(_shared_embedding_information['class_node'].values())).shape[-1]
        features = torch.zeros(num_nodes, embed_dim)
        for _, row in nodes_df.iterrows():
            old_node_id = int(row['node_id'])
            new_node_id = int(row['new_id'])
            ifc_class = row['item']
            if old_node_id == 0:
                feat = _shared_embedding_information['class_node'].get('IfcElement')
            else:
                feat = _shared_embedding_information['class_node'].get(ifc_class)
            if feat is None:
                raise ValueError(f'IfcClass {ifc_class} not in embedding table | node_id={old_node_id} | file={node_file_path}')
            features[new_node_id] = feat

        # 只保留两端都在节点集合中的边
        valid_mask = edges_df['src_id'].isin(id_map) & edges_df['dst_id'].isin(id_map)
        edges_df = edges_df[valid_mask].copy()
        if len(edges_df) > 0:
            src = torch.tensor(edges_df['src_id'].map(id_map).to_numpy(), dtype=torch.long)
            dst = torch.tensor(edges_df['dst_id'].map(id_map).to_numpy(), dtype=torch.long)
            hop = torch.tensor(edges_df['hop'].to_numpy(), dtype=torch.long)
        else:
            src = torch.zeros(0, dtype=torch.long)
            dst = torch.zeros(0, dtype=torch.long)
            hop = torch.zeros(0, dtype=torch.long)

        # 构图
        g = dgl.graph((dst, src), num_nodes=num_nodes)
        g.edata['hop'] = hop
        g.ndata['features'] = features
        # 自环
        self_ids = torch.arange(num_nodes, dtype=torch.long)
        g = dgl.add_edges(g, self_ids, self_ids, data={'hop': torch.zeros(num_nodes, dtype=torch.long)})

        y = self.class_to_idx[label]
        return model_flag, g, y

    # =======================================
    # build 始终构建三张图的超集，按三者交集对齐
    # =======================================
    def process(self):
        rep_path_list, spa_path_list, sem_path_list = [], [], []
        for label in os.listdir(self.root):
            type_path = os.path.join(self.root, label)
            for item in os.listdir(str(type_path)):
                rep_path = os.path.normpath(os.path.join(str(type_path), str(item), "GRAPH"))
                ctx_path = os.path.normpath(os.path.join(str(type_path), str(item), "CONTEXT"))
                rep_path_list.append((label, rep_path))
                spa_path_list.append((label, ctx_path))
                sem_path_list.append((label, ctx_path))

        rep_map = self._flatten_results(self._run_multiprocess(rep_path_list, self._process_rep_label_path, tag="REP"))
        spa_map = self._flatten_results(self._run_multiprocess(spa_path_list, self._process_spa_label_path, tag="SPA"))
        sem_map = self._flatten_results(self._run_multiprocess(sem_path_list, self._process_sem_label_path, tag="SEM"))

        common_keys = sorted(set(rep_map) & set(spa_map) & set(sem_map))
        if len(common_keys) == 0:
            raise RuntimeError(f"REP/SPA/SEM 三者无公共 model_flag: rep={len(rep_map)} spa={len(spa_map)} sem={len(sem_map)}")

        for tag, a, b in [("REP-SPA", rep_map, spa_map), ("REP-SEM", rep_map, sem_map), ("SPA-SEM", spa_map, sem_map)]:
            miss = sorted(set(a) - set(b))
            if miss:
                print(f"[BIMContextNetDataset] WARNING: {len(miss)} {tag} 缺失对齐 (show 5): {miss[:5]}")

        self.model_flag, self.rep_graphs, self.spa_graphs, self.sem_graphs, self.labels = [], [], [], [], []
        for k in common_keys:
            rep_g, rep_y = rep_map[k]
            spa_g, spa_y = spa_map[k]
            sem_g, sem_y = sem_map[k]
            if not (int(rep_y) == int(spa_y) == int(sem_y)):
                raise ValueError(f"label mismatch at {k}: rep={int(rep_y)} spa={int(spa_y)} sem={int(sem_y)}")
            self.model_flag.append(k)
            self.rep_graphs.append(rep_g)
            self.spa_graphs.append(spa_g)
            self.sem_graphs.append(sem_g)
            self.labels.append(rep_y)

        print(f"[BIMContextNetDataset] aligned samples = {len(self.model_flag)} (split={self.split})")

    # ---------
    # caching
    # ----------
    def _paths(self):
        base = os.path.join(self._cache_path, self.split)
        return {
            'rep': os.path.join(base, MODALITY_BIN['rep']),
            'spa': os.path.join(base, MODALITY_BIN['spa']),
            'sem': os.path.join(base, MODALITY_BIN['sem']),
            'label': os.path.join(base, 'fusion_labels.pt'),
            'flag': os.path.join(base, 'fusion_flags.pt'),
        }

    def save(self):
        base = os.path.join(self._cache_path, self.split)
        os.makedirs(base, exist_ok=True)
        p = self._paths()
        # 排序稳态
        sorted_data = sorted(
            zip(self.model_flag, self.rep_graphs, self.spa_graphs, self.sem_graphs, self.labels),
            key=lambda x: x[0])
        self.model_flag, self.rep_graphs, self.spa_graphs, self.sem_graphs, self.labels = map(list, zip(*sorted_data))
        save_graphs(p['rep'], self.rep_graphs)
        save_graphs(p['spa'], self.spa_graphs)
        save_graphs(p['sem'], self.sem_graphs)
        torch.save(self.labels, p['label'])
        torch.save(self.model_flag, p['flag'])

    def load(self):
        p = self._paths()
        # 只读被请求模态的 bin
        self.rep_graphs = load_graphs(p['rep'])[0] if 'rep' in self.modalities else []
        self.spa_graphs = load_graphs(p['spa'])[0] if 'spa' in self.modalities else []
        self.sem_graphs = load_graphs(p['sem'])[0] if 'sem' in self.modalities else []
        self.labels = torch.load(p['label'], weights_only=True)
        self.model_flag = torch.load(p['flag'], weights_only=True)

    def has_cache(self):
        # 缓存是三张图的超集
        p = self._paths()
        return all(os.path.exists(p[k]) for k in ('rep', 'spa', 'sem', 'label', 'flag'))

    # ------------
    # dataset api
    # ------------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {}

        if 'rep' in self.modalities:
            g = self.rep_graphs[index]
            if self.edge_direction == 'REVERSE':
                g = dgl.reverse(g, copy_ndata=True, copy_edata=True)
            sample['rep'] = g

        if 'sem' in self.modalities:
            g = self.sem_graphs[index]
            if self.edge_direction == 'REVERSE':
                g = dgl.reverse(g, copy_ndata=True, copy_edata=True)
            sample['sem'] = g

        if 'spa' in self.modalities:
            g = self.spa_graphs[index]
            # 读取期按 n_hops 现裁
            if 'hop' in g.edata:
                keep = (g.edata['hop'] <= self.n_hops)
                g = dgl.edge_subgraph(g, keep, relabel_nodes=False)
            sample['spa'] = g

        sample['label'] = int(self.labels[index])
        sample['flag'] = self.model_flag[index]
        return sample


def collate_modalities(samples):

    present = [m for m in VALID_MODALITIES if m in samples[0]]
    batch = {m: dgl.batch([s[m] for s in samples]) for m in present}
    batch['label'] = torch.tensor([int(s['label']) for s in samples], dtype=torch.long)
    batch['flag'] = [s['flag'] for s in samples]
    return batch