from proteins.runs.data import get_dataset_cluster_loaders


def main():
    train_loader, test_loader, sizes = get_dataset_cluster_loaders(
        train_parts=40 * 16 * 4, test_parts=8 * 16 * 8
    )

    for i, data in enumerate(train_loader):
        print(i, data.x.shape, data.edge_index.shape, data.edge_attr.shape)
        if i == 490:
            print("found")

    for i, data in enumerate(train_loader):
        print(i, data.x.shape, data.edge_index.shape, data.edge_attr.shape)


if __name__ == "__main__":
    main()
