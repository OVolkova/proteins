import lightning as L
import torch
from ogb.nodeproppred import Evaluator


class TrainerProteins(L.LightningModule):
    def __init__(
        self,
        models,
        training_config,
    ):
        super().__init__()

        self.model = models[0]
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator("ogbn-proteins")
        self.training_config = training_config

        self.test_y_true = {"train": [], "valid": [], "test": []}
        self.test_y_pred = {"train": [], "valid": [], "test": []}

        self.save_hyperparameters()

    def forwar(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch.x.shape[0] == 0:
            return
        data = batch.to(self.device)
        out, _ = self.model(data.x, data.edge_index, data.edge_attr, device=self.device)
        loss = self.loss(out[data.train_mask], data.y[data.train_mask])
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch.x.shape[0] == 0:
            return
        data = batch.to(self.device)
        out, _ = self.model(data.x, data.edge_index, data.edge_attr, device=self.device)

        for split in self.test_y_true.keys():
            mask = data[f"{split}_mask"]
            self.test_y_true[split].append(data.y[mask].detach())
            self.test_y_pred[split].append(out[mask].detach())

    def roc_auc_for_split(self, split):
        if torch.cat(self.test_y_true[split], dim=0).size()[0] > 0:
            train_rocauc = self.evaluator.eval(
                {
                    "y_true": torch.cat(self.test_y_true[split], dim=0),
                    "y_pred": torch.cat(self.test_y_pred[split], dim=0),
                }
            )["rocauc"]
            self.log(split + "_rocauc", train_rocauc)

        self.test_y_true[split] = []
        self.test_y_pred[split] = []

    def on_validation_epoch_end(self):
        self.roc_auc_for_split("train")
        self.roc_auc_for_split("valid")
        self.roc_auc_for_split("test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
            eps=self.training_config.eps,
        )
