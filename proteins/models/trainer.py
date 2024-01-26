import pytorch_lightning as pl
import torch
from ogb.nodeproppred import Evaluator


class TrainerProteins(pl.LightningModule):
    def __init__(
        self,
        model_class,
        model_config,
        training_config,
    ):
        super().__init__()

        self.model = model_class(model_config)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator("ogbn-proteins")
        self.training_config = training_config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data = batch.to(self.device)
        out = self.model(data.x, data.edge_index, data.edge_attr)
        loss = self.loss(out[data.train_mask], data.y[data.train_mask])
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_true = {"train": [], "valid": [], "test": []}
        y_pred = {"train": [], "valid": [], "test": []}

        data = batch.to(self.device)
        out = self.model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f"{split}_mask"]
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        train_rocauc = self.evaluator.eval(
            {
                "y_true": torch.cat(y_true["train"], dim=0),
                "y_pred": torch.cat(y_pred["train"], dim=0),
            }
        )["rocauc"]

        valid_rocauc = self.evaluator.eval(
            {
                "y_true": torch.cat(y_true["valid"], dim=0),
                "y_pred": torch.cat(y_pred["valid"], dim=0),
            }
        )["rocauc"]

        test_rocauc = self.evaluator.eval(
            {
                "y_true": torch.cat(y_true["test"], dim=0),
                "y_pred": torch.cat(y_pred["test"], dim=0),
            }
        )["rocauc"]

        self.log("train_rocauc", train_rocauc)
        self.log("valid_rocauc", valid_rocauc)
        self.log("test_rocauc", test_rocauc)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
            eps=self.training_config.eps,
        )
