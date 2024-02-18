from proteins.models.base_models.deepergcn import DeeperGCN
from proteins.models.config import GraphTransformerConfig, TrainingConfig
from proteins.runs.data import get_dataset_loaders, get_dataset_cluster_loaders
from proteins.runs.runs import run_training
from proteins.models.decoders import FeaturesPredictor, LinkPredictor
from proteins.models.pretrain import GraphPreTrainer


def main():
    # train_loader, test_loader, sizes = get_dataset_loaders(train_parts=40, test_parts=8)
    train_loader, test_loader, sizes = get_dataset_cluster_loaders(
        train_parts=40 * 16 * 4,
        test_parts=8 * 16 * 16,
        train_batch_size=16,
        test_batch_size=8,
    )
    model_config = GraphTransformerConfig(
        d_node_in=sizes["node_in"],
        d_edge_in=sizes["edge_in"],
        d_node_out=64,
        d_embed=64,
        n_layers=28,
        attention_dropout=0.1,
    )
    model_config.loader = "cluster 40 * 16 * 4"
    model_config.pretrain = True

    run_training(
        trainer_class=GraphPreTrainer,
        model_config=model_config,
        training_config=TrainingConfig,
        models=[
            DeeperGCN(model_config),
            FeaturesPredictor(
                model_config.d_node_out, sizes["node_in"], model_config.linear_dropout
            ),
            LinkPredictor(
                model_config.d_node_out, sizes["edge_in"], model_config.linear_dropout
            ),
            LinkPredictor(model_config.d_node_out, 2, model_config.linear_dropout),
        ],
        train_loader=train_loader,
        test_loader=test_loader,
        run_name="DeeperGCN",
        project="OGBN-Proteins-Pretrain",
        model_path=None,
    )


if __name__ == "__main__":
    main()
