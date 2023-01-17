import math

import torch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, Set2Set, JumpingKnowledge, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, dense_diff_pool, ASAPooling
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
import torch.nn.functional as F

from layers import SGConv, SSGConv, CTGConv, GINConvWeight, SAGEWeightConv, Block


# from basic_layers import GCNWeightConv


class Net_GCN(torch.nn.Module):
    def __init__(self, args):
        super(Net_GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu((self.conv1(x, edge_index)))
        x = F.relu((self.conv2(x, edge_index)))
        x = F.relu((self.conv3(x, edge_index)))

        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        # x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_GCNWeight(torch.nn.Module):
    def __init__(self, args):
        super(Net_GCNWeight, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = F.relu((self.conv1(x, edge_index, edge_weight)))
        x = F.relu((self.conv2(x, edge_index, edge_weight)))
        x = F.relu((self.conv3(x, edge_index, edge_weight)))

        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        # x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_GAT(torch.nn.Module):
    def __init__(self, args):
        super(Net_GAT, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.heads = 8

        self.conv1 = GATConv(self.num_features, self.nhid, self.heads, dropout=self.dropout_ratio)
        self.conv2 = GATConv(self.nhid * self.heads, self.nhid * self.heads, dropout=self.dropout_ratio)
        self.conv3 = GATConv(self.nhid * self.heads, self.nhid * self.heads, dropout=self.dropout_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * self.heads, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu((self.conv1(x, edge_index)))
        x = F.elu((self.conv2(x, edge_index)))
        x = F.elu((self.conv3(x, edge_index)))

        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        # x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_GIN(torch.nn.Module):
    def __init__(self, args):
        super(Net_GIN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GINConv(
            Sequential(Linear(self.num_features, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_GIN_W(torch.nn.Module):
    def __init__(self, args):
        super(Net_GIN_W, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GINConvWeight(
            Sequential(Linear(self.num_features, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv2 = GINConvWeight(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.conv3 = GINConvWeight(
            Sequential(Linear(self.nhid, self.nhid), BatchNorm1d(self.nhid), ReLU(),
                       Linear(self.nhid, self.nhid), ReLU()))

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.conv3(x, edge_index, edge_weight)

        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_SGC(torch.nn.Module):
    def __init__(self, args):
        super(Net_SGC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.K = args.K

        self.conv1 = SGConv(self.num_features, self.nhid, K=self.K, cached=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = (self.conv1(x, edge_index, edge_weight))

        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        # x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_SSGC(torch.nn.Module):
    def __init__(self, args):
        super(Net_SSGC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.K = args.K

        self.conv1 = SSGConv(self.num_features, self.nhid, K=self.K, cached=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = (self.conv1(x, edge_index))

        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        # x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class Net_CTGC(torch.nn.Module):
    def __init__(self, args):
        super(Net_CTGC, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.K = args.K

        self.alpha = args.alpha
        self.aggr_type = args.aggr_type
        self.norm_type = args.norm_type

        self.conv1 = CTGConv(self.num_features, self.nhid, K=self.K, alpha=self.alpha, aggr_type=args.aggr_type,
                             norm_type=args.norm_type, cached=False)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch

        x = (self.conv1(x, edge_index, edge_weight))

        # x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        x = global_add_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = SAGEConv(self.num_features, self.nhid)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.nhid, self.nhid))
        self.lin1 = Linear(self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGEWeight(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = SAGEWeightConv(self.num_features, self.nhid)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEWeightConv(self.nhid, self.nhid))
        self.lin1 = Linear(self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Net_Set2Set(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = SAGEConv(self.num_features, self.nhid)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(self.nhid, self.nhid))
        self.set2set = Set2Set(self.nhid, processing_steps=4)
        self.lin1 = Linear(2 * self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.set2set.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.set2set(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Net_Set2SetWeight(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = SAGEWeightConv(self.num_features, self.nhid)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEWeightConv(self.nhid, self.nhid))
        self.set2set = Set2Set(self.nhid, processing_steps=4)
        self.lin1 = Linear(2 * self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.set2set.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        x = self.set2set(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class DiffPool(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.ratio = 0.25
        self.num_nodes = math.ceil(args.num_nodes_pool * self.ratio)

        self.embed_block1 = Block(self.num_features, self.nhid, self.nhid)
        self.pool_block1 = Block(self.num_features, self.nhid, self.num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((self.num_layers // 2) - 1):
            num_nodes = math.ceil(self.ratio * self.num_nodes)
            self.embed_blocks.append(Block(self.nhid, self.nhid, self.nhid))
            self.pool_blocks.append(Block(self.nhid, self.nhid, self.num_nodes))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((len(self.embed_blocks) + 1) * self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.pool_block1(x, adj, mask)
        x = F.relu(self.embed_block1(x, adj, mask))
        xs = [x.mean(dim=1)]
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Net_ASAP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.ratio = 0.25

        self.conv1 = GraphConv(self.num_features, self.nhid, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(self.nhid, self.nhid, aggr='mean')
            for _ in range(self.num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(self.nhid, self.ratio, dropout=self.dropout_ratio)
            for _ in range(self.num_layers // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(self.num_layers * self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Net_ASAPWeight(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.ratio = 0.25

        self.conv1 = GraphConv(self.num_features, self.nhid, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(self.nhid, self.nhid, aggr='mean')
            for _ in range(self.num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(self.nhid, self.ratio, dropout=self.dropout_ratio)
            for _ in range(self.num_layers // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(self.num_layers * self.nhid, self.nhid)
        self.lin2 = Linear(self.nhid, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_weight
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
