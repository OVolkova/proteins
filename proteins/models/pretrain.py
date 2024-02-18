import lightning as L
import torch
from torch_geometric.utils import negative_sampling
from torchmetrics import AveragePrecision, MeanSquaredError
from torchmetrics import MetricCollection
from proteins.models.scaled_loss import ScaledLoss


class GraphPreTrainer(L.LightningModule):
    def __init__(
        self,
        models,
        training_config,
        mask_probability=0.3,
    ):
        super().__init__()
        # Define models
        self.encoder, self.node_decoder, self.edge_decoder, self.link_prediction = (
            models
        )

        # Define loss functions
        self.loss_nodes = torch.nn.MSELoss()
        self.loss_edge_attr = torch.nn.MSELoss()
        self.loss_links = torch.nn.CrossEntropyLoss()
        self.multi_task_loss = ScaledLoss(
            weights=torch.tensor([1, 1, 1], device=self.device), device=self.device
        )

        # Define metrics
        self.splits = ["train", "valid"]
        self.nodes_mse = {split: MeanSquaredError() for split in self.splits}
        self.edge_attr_mse = {split: MeanSquaredError() for split in self.splits}
        self.link_prediction_ap = {
            split: AveragePrecision(task="binary") for split in self.splits
        }

        # Define training_config
        self.training_config = training_config
        self.mask_probability = mask_probability

        self.save_hyperparameters()

    def mask(self, x):
        mask = torch.rand(x.size(), device=self.device) < self.mask_probability
        masked_x = x.clone()
        masked_x = masked_x.masked_fill(mask, 0)
        return mask, masked_x

    def forward(self, batch, run_type="train"):
        if batch.x.shape[0] == 0:
            return

        data = batch.to(self.device)

        # Mask input data
        mask_x, masked_x = self.mask(data.x)
        mask_e, masked_edge_attr = self.mask(data.edge_attr)

        if run_type == "valid":
            mask_x_v, masked_x_v = self.mask(data.x[data["valid_mask"]])
            masked_x[data["valid_mask"],:] = masked_x_v
            mask_x[data["valid_mask"],:] = mask_x_v

            valid_edge_index = data["valid_mask"][data.edge_index[0]]
            mask_e_v, masked_e_v = self.mask(data.edge_attr[valid_edge_index])
            masked_edge_attr[valid_edge_index,:] = masked_e_v
            mask_e[valid_edge_index,:] = mask_e_v

        # Encode masked input data
        x_hat, e_hat = self.encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            device=self.device,
        )

        # predict the masked values of edges features
        if e_hat is not None:
            e_hat = self.edge_decoder(e_hat)
        else:
            e_hat = self.edge_decoder(x_hat, data.edge_index)

        # Compute link prediction with positive and negative sampling
        negative = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
        positive = data.edge_index
        links = torch.cat([positive, negative], dim=-1)
        links_hat = self.link_prediction(x_hat, links)
        links_true = torch.cat(
            [
                torch.ones(positive.size(1), dtype=torch.long, device=self.device),
                torch.zeros(negative.size(1), dtype=torch.long, device=self.device),
            ],
            dim=-1,
        )

        # predict the masked values of nodes features
        x_hat = self.node_decoder(x_hat)

        if run_type == "train":
            # Compute loss only for training split
            loss_nodes = self.loss_nodes(
                torch.masked_select(x_hat, mask_x), torch.masked_select(data.x, mask_x)
            )
            loss_edge_attr = self.loss_edge_attr(
                torch.masked_select(e_hat, mask_e),
                torch.masked_select(data.edge_attr, mask_e),
            )
            loss_links = self.loss_links(links_hat, links_true)

            loss = self.multi_task_loss(
                torch.stack([loss_nodes, loss_edge_attr, loss_links])
            )

            self.log_dict(
                {
                    "loss/total": loss,
                    "loss/nodes": loss_nodes,
                    "loss/edge_attr": loss_edge_attr,
                    "loss/links_prediction": loss_links,
                }
            )
            return loss
        else:
            # Compute metrics if not training split
            for split in self.splits:
                mask_x_split = mask_x.masked_fill(
                    ~data[f"{split}_mask"].view(-1, 1).expand(mask_x.size()), False
                )
                mask_e_split = mask_e.masked_fill(
                    ~ data[f"{split}_mask"][data.edge_index[0]]
                    .view(-1, 1)
                    .expand(mask_e.size()),
                    False,
                )
                self.nodes_mse[split].update(
                    torch.masked_select(x_hat, mask_x_split),
                    torch.masked_select(data.x, mask_x_split),
                )
                self.edge_attr_mse[split].update(
                    torch.masked_select(e_hat, mask_e_split),
                    torch.masked_select(data.edge_attr, mask_e_split),
                )
                self.link_prediction_ap[split].update(
                        links_hat.T[1][data[f"{split}_mask"][links[0]]],
                        links_true[data[f"{split}_mask"][links[0]]]
                    )

    def training_step(self, batch, batch_idx):
        return self.forward(batch, run_type="train")

    def validation_step(self, batch, batch_idx):
        _ = self.forward(batch, run_type="valid")

    def metrics_for_split(self, split):
        self.log_dict(
            {
                split + "/nodes_mse": self.nodes_mse[split].compute(),
                split + "/edge_attr_mse": self.edge_attr_mse[split].compute(),
                split + "/link_prediction_ap": self.link_prediction_ap[split].compute(),
            }
        )

        self.nodes_mse[split].reset()
        self.edge_attr_mse[split].reset()
        self.link_prediction_ap[split].reset()

    def on_validation_epoch_end(self):
        for split in self.splits:
            self.metrics_for_split(split)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
            eps=self.training_config.eps,
        )

    def on_validation_epoch_start(self) -> None:
        for split in self.splits:
            if self.nodes_mse[split].device != self.device:
                self.nodes_mse[split] = self.nodes_mse[split].to(self.device)
                self.edge_attr_mse[split] = self.edge_attr_mse[split].to(self.device)
                self.link_prediction_ap[split] = self.link_prediction_ap[split].to(
                    self.device
                )

    def on_train_epoch_start(self) -> None:
        if self.multi_task_loss.a.device != self.device:
            self.multi_task_loss.move_to_device(self.device)
