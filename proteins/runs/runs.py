import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger


def run_training(
    trainer_class,
    model_config,
    training_config,
    models,
    train_loader,
    test_loader,
    project="OGBN-Proteins",
    run_name="unknown",
    model_path=None,
):
    torch.set_float32_matmul_precision("medium")

    if model_path is not None:
        model_ = trainer_class.load_from_checkpoint(
            checkpoint_path=model_path,
            models=models,
            training_config=training_config,
        )
    else:
        model_ = trainer_class(
            models=models,
            training_config=training_config,
        )

    wandb_logger = WandbLogger(project=project, name=run_name)
    wandb_logger.experiment.config.update(model_config.__dict__)
    wandb_logger.experiment.config.update(
        {k: v for k, v in training_config.__dict__.items() if not k.startswith("__")}
    )
    wandb_logger.experiment.config.update(
        {"gradient_clip_val": 0.5, "gradient_clip_algorithm": "value"}
    )

    trainer = L.Trainer(
        max_epochs=500,
        log_every_n_steps=1,
        logger=wandb_logger,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
    )
    trainer.fit(
        model=model_, train_dataloaders=train_loader, val_dataloaders=test_loader
    )
