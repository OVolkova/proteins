import pytorch_lightning as pl

from proteins.models.config import GraphTransformerConfig, TrainingConfig
from proteins.models.base_models.deepergcn import DeeperGCN
from proteins.models.trainer import TrainerProteins
from proteins.runs.data import get_dataset_loaders

if __name__ == "__main__":
    train_loader, test_loader, sizes = get_dataset_loaders()

    model_config = GraphTransformerConfig(
        d_node_in=sizes["node_in"],
        d_edge_in=sizes["edge_in"],
        d_node_out=sizes["node_out"],
        d_embed=64,
        n_layers=28,
        attention_dropout=0.1,
    )

    model_ = TrainerProteins(
        model_class=DeeperGCN, model_config=model_config, training_config=TrainingConfig
    )
    # model_ = TrainingModel._load_from_checkpoint(
    #     "lightning_logs/version_90/checkpoints/epoch=499-step=312500.ckpt"
    # )

    trainer = pl.Trainer(
        max_epochs=500,
        log_every_n_steps=1,
    )

    trainer.fit(model=model_, train_dataloaders=train_loader, val_dataloaders=test_loader)
