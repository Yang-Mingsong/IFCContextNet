import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn



def custom_aggregate(inputs, dsttype):
    if dsttype == 'class_node':
        part1 = inputs[0]
        part2 = torch.zeros_like(inputs[0])
        part3 = inputs[1]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'attribute_node' or dsttype == 'type_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.max(torch.stack([inputs[0], inputs[1], inputs[2]]), dim=0)[0]
        part3 = inputs[3]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    elif dsttype == 'value_node':
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        part3 = inputs[0]
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)
    else:
        print("______大错特错")
        part1 = torch.zeros_like(inputs[0])
        part2 = torch.zeros_like(inputs[0])
        part3 = torch.zeros_like(inputs[0])
        aggregated_feats = torch.cat((part1, part2, part3), dim=1)

    return aggregated_feats


class RGCNEncoder(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super(RGCNEncoder, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate=custom_aggregate)
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(3 * hid_feats, out_feats)
            for rel in rel_names}, aggregate=custom_aggregate)

        self.fc1 = nn.Linear(12 * out_feats, 3 * out_feats)
        self.bn1 = nn.BatchNorm1d(3 * out_feats)
        self.fc2 = nn.Linear(3 * out_feats, out_feats)
        self.bn2 = nn.BatchNorm1d(out_feats)

    def forward(self, graph):
        # inputs是节点的特征
        h = graph.ndata['features']
        h = self.conv1(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}

        with graph.local_scope():
            graph.ndata['features'] = h
            features_list = []
            for ntype in graph.ntypes:
                features_list.append(dgl.max_nodes(graph, 'features', ntype=ntype))
            rep_hg = torch.cat(features_list, dim=1)
            rep_hg = F.relu(self.bn1(self.fc1(rep_hg)))
            rep_hg = F.relu(self.bn2(self.fc2(rep_hg)))
        return rep_hg


class RGATSemEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names, n_hops=3,
                 num_heads=1, feat_drop=0.2, attn_drop=0.2):
        super(RGATSemEncoder, self).__init__()
        head_dim = hidden_dim // num_heads

        def make_conv(in_f, activation=None):
            return dglnn.HeteroGraphConv(
                {rel: dglnn.GATConv(in_f, head_dim, num_heads=num_heads,
                                    feat_drop=feat_drop, attn_drop=attn_drop,
                                    allow_zero_in_degree=True,
                                    activation=activation)
                 for rel in rel_names},
                aggregate="sum"
            )

        self.layers = nn.ModuleList()
        if n_hops == 1:
            self.layers.append(make_conv(in_dim))
        else:
            self.layers.append(make_conv(in_dim, activation=F.relu))
            for _ in range(n_hops - 2):
                self.layers.append(make_conv(hidden_dim, activation=F.relu))
            self.layers.append(make_conv(hidden_dim))

        self.fc = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

        # 新增:空 sem 的 fallback 投影,模拟 GAT 层的非线性变换
        self.empty_projection = nn.Linear(in_dim, hidden_dim)

    def forward(self, g):
        h = g.ndata['features']

        # 提前抠出 target 初始嵌入(空 sem fallback 用)
        init_class = h['class_node']
        offsets = torch.zeros(g.batch_size, dtype=torch.long, device=init_class.device)
        offsets[1:] = g.batch_num_nodes('class_node').cumsum(0)[:-1]
        target_init = init_class[offsets]  # [B, in_dim]

        with g.local_scope():
            for layer in self.layers:
                h = layer(g, h)
                h = {k: v.flatten(1) for k, v in h.items()}

            # GNN 路径的 target 表征。class_node 缺失(A/C 空 sem)时占位即可,
            # 反正下面 torch.where 会被 is_empty 覆盖掉
            if 'class_node' in h and h['class_node'].shape[0] == g.num_nodes('class_node'):
                target_gnn = h['class_node'][offsets]  # [B, hidden_dim]
            else:
                target_gnn = self.empty_projection(target_init)  # 占位

            # 空 sem fallback 路径
            target_fallback = self.empty_projection(target_init)  # [B, hidden_dim]

            # 空 sem 检测
            is_empty = (g.batch_num_nodes('attribute_node') == 0)  # [B] bool

            # 按样本选择
            target_feat = torch.where(is_empty.unsqueeze(-1), target_fallback, target_gnn)

            out = F.relu(self.bn(self.fc(target_feat)))
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_hops):
        super(GCNEncoder, self).__init__()

        self.layers = nn.ModuleList()

        if n_hops == 1:
            self.layers.append(dglnn.GraphConv(in_dim, hidden_dim))
        else:
            self.layers.append(dglnn.GraphConv(in_dim, hidden_dim, activation=nn.ReLU()))
            for _ in range(n_hops - 2):
                self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU()))
            self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, g):
        h = g.ndata['features']
        with g.local_scope():
            for layer in self.layers:
                h = layer(g, h)


            # batch_num_nodes() = [n_0, n_1, ..., n_{B-1}]
            # 每个子图 node-0 在 batched h 里的全局索引 = [0, n_0, n_0+n_1, ...]
            offsets = torch.zeros(g.batch_size, dtype=torch.long, device=h.device)
            offsets[1:] = torch.cumsum(g.batch_num_nodes(), 0)[:-1]
            geo_hg = h[offsets]  # [B, hidden_dim]
            # ====================================

            # g.ndata['h'] = h
            # geo_hg = dgl.mean_nodes(g, 'h')

            geo_hg = F.relu(self.bn(self.fc(geo_hg)))

        return geo_hg


class SemAvailabilityEncoder(nn.Module):
    def __init__(self, out_dim):
        super(SemAvailabilityEncoder, self).__init__()
        self.embedding = nn.Embedding(2, out_dim)

    def forward(self, g):
        avail = (g.batch_num_nodes('attribute_node') > 0).long()
        return self.embedding(avail)