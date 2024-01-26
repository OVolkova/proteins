import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.utils import scatter


def get_dataset_loaders():
    dataset = PygNodePropPredDataset("ogbn-proteins", root="../data")
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.node_species = None
    data.y = data.y.to(torch.float)
    # Initialize features of nodes by aggregating edge features.
    row, col = data.edge_index
    data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce="sum")

    # Set split indices to masks.
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f"{split}_mask"] = mask

    train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True, num_workers=5)
    test_loader = RandomNodeLoader(data, num_parts=5, num_workers=5)

    sizes = {
        "node_in": data.x.size(-1),
        "edge_in": data.edge_attr.size(-1),
        "node_out": data.y.size(-1),
    }
    return train_loader, test_loader, sizes
