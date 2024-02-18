from proteins.models.config import GraphTransformerConfig, TrainingConfig
from proteins.models.graph_sparse.model import GraphTransformer
from proteins.runs.data import get_dataset_cluster_loaders
from proteins.runs.runs import run_training
from proteins.models.decoders import FeaturesPredictor, LinkPredictor
from proteins.models.pretrain import GraphPreTrainer


def main():
    train_loader, test_loader, sizes = get_dataset_cluster_loaders(
        train_parts=40 * 16 * 4, test_parts=8 * 16 * 16
    )

    model_config = GraphTransformerConfig(
        d_node_in=sizes["node_in"],
        d_edge_in=sizes["edge_in"],
        d_node_out=16,
        d_edge_out=16,
        d_embed=16,
        n_heads=8,
        n_layers=32,
        d_ff=32,
        # # graph specific part of the config starts here
        d_e_embed=16,
        e_heads=8,
        d_e_ff=32,
        simple_attention=True,
    )

    run_training(
        trainer_class=GraphPreTrainer,
        model_config=model_config,
        training_config=TrainingConfig,
        models=[
            GraphTransformer(model_config),
            FeaturesPredictor(
                model_config.d_embed, sizes["node_in"], model_config.linear_dropout
            ),
            FeaturesPredictor(
                model_config.d_e_embed, sizes["edge_in"], model_config.linear_dropout
            ),
            LinkPredictor(model_config.d_embed, 2, model_config.linear_dropout),
        ],
        train_loader=train_loader,
        test_loader=test_loader,
        run_name="Attention2n",
        project="OGBN-Proteins-Pretrain",
        # model_path="/home/oskoshcheeva/projects/proteins/proteins/runs/OGBN-Proteins/nno2jbb2/checkpoints/epoch=37-step=12160.ckpt",
    )


if __name__ == "__main__":
    main()
