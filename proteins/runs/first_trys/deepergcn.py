from proteins.models.base_models.deepergcn import DeeperGCN
from proteins.models.config import GraphTransformerConfig, TrainingConfig
from proteins.runs.data import get_dataset_loaders, get_dataset_cluster_loaders
from proteins.runs.runs import run_training
from proteins.models.trainer import TrainerProteins


def main():
    # train_loader, test_loader, sizes = get_dataset_loaders(train_parts=40, test_parts=8)
    train_loader, test_loader, sizes = get_dataset_cluster_loaders(
        train_parts=40 * 16 * 4, test_parts=8 * 16 * 16
    )
    model_config = GraphTransformerConfig(
        d_node_in=sizes["node_in"],
        d_edge_in=sizes["edge_in"],
        d_node_out=sizes["node_out"],
        d_embed=64,
        n_layers=2,
        attention_dropout=0.1,
    )
    model_config.loader = "cluster 40 * 16 * 4"

    run_training(
        trainer_class=TrainerProteins,
        model_config=model_config,
        training_config=TrainingConfig,
        models=[DeeperGCN(model_config)],
        train_loader=train_loader,
        test_loader=test_loader,
        run_name="DeeperGCN",
        model_path=None,
    )


if __name__ == "__main__":
    main()
