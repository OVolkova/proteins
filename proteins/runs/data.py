import os

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.loader.cluster import ClusterData, ClusterLoader
from torch_geometric.utils import scatter
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from proteins.logger import logging


def get_subgraph(data: Data, subgraph_nodes: torch.Tensor):
    logging.info("selecting subgraph")
    edge_index, edge_attr = subgraph(subgraph_nodes, data.edge_index, data.edge_attr)

    subgraph_data = Data(
        x=data.x[subgraph_nodes],
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data.y[subgraph_nodes],
        num_nodes=subgraph_nodes.size(0),
    )
    # Set split indices to masks.
    for split in ["train", "valid", "test"]:
        subgraph_data[f"{split}_mask"] = data[f"{split}_mask"][subgraph_nodes]
    logging.info("done selecting subgraph")
    return subgraph_data


def read_dataset():
    dataset = PygNodePropPredDataset(
        "ogbn-proteins", root=os.path.join(get_path_to_save(), "data")
    )
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)
    # Initialize features of nodes by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce="sum")
    data.x = torch.log(data.x + 1)

    # Set split indices to masks.
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f"{split}_mask"] = mask

    return data, split_idx


def get_graph_sizes(data):
    return {
        "node_in": data.x.size(-1),
        "edge_in": data.edge_attr.size(-1),
        "node_out": data.y.size(-1),
    }


def get_path_to_save():
    for i, p in enumerate(__file__.split("/")[::-1]):
        if p == "proteins":
            break
    return "/".join(__file__.split("/")[: -i - 1])


def get_dataset_loaders(train_parts=40, test_parts=8):
    data, _ = read_dataset()

    train_loader = RandomNodeLoader(
        data, num_parts=train_parts, shuffle=True, num_workers=2, pin_memory=False
    )
    test_loader = RandomNodeLoader(
        data, num_parts=test_parts, num_workers=2, pin_memory=False
    )

    return train_loader, test_loader, get_graph_sizes(data)


def get_dataset_cluster_loaders(
    train_parts=40, test_parts=8, train_batch_size=4, test_batch_size=1
):
    data, split_idx = read_dataset()

    nodes = torch.arange(data["num_nodes"])
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    path = os.path.join(os.path.join(get_path_to_save(), "data"), "cluster")
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, "train"))
        os.mkdir(os.path.join(path, "valid"))

    logging.info("cluster_data")
    mask[split_idx["train"]] = True
    train_cluster_data = ClusterData(
        data=get_subgraph(data, nodes[mask]),
        num_parts=train_parts,
        save_dir=os.path.join(path, "train"),
    )
    train_loader = ClusterLoader(
        train_cluster_data, batch_size=train_batch_size, shuffle=True
    )
    logging.info("done cluster_data")

    logging.info("cluster_data")
    mask[split_idx["valid"]] = True
    valid_cluster_data = ClusterData(
        data=get_subgraph(data, nodes[mask]),
        num_parts=test_parts,
        save_dir=os.path.join(path, "valid"),
    )
    valid_loader = ClusterLoader(
        valid_cluster_data, batch_size=test_batch_size, shuffle=False
    )
    logging.info("done cluster_data")
    return train_loader, valid_loader, get_graph_sizes(data)
