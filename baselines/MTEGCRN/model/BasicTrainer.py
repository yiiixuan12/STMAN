import torch
import math
import os
import time
import copy
import json
from pathlib import Path
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader,
                 test_loader, scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir,
                                            'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        # log
        if not os.path.isdir(args.log_dir) and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model,
                                  debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.train_losses = []
        self.val_losses = []

    def _batch_to_device(self, data, target):
        data = data.to(self.args.device, non_blocking=True)
        target = target.to(self.args.device, non_blocking=True)
        return data, target

    def val_epoch(self, epoch, val_dataloader, i=2):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data, target = self._batch_to_device(data, target)
                label = target[..., :self.args.output_dim]
                output = self.model(data, i)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                loss = self.loss(output, label)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info(
            '***********Val Epoch {}: average Loss: {:.6f}'.format(
                epoch, val_loss))
        self.val_losses.append(val_loss)
        return val_loss

    def train_epoch(self, epoch, i=2):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = self._batch_to_device(data, target)
            label = target[..., :self.args.output_dim]
            self.optimizer.zero_grad()
            output = self.model(data, i)
            if self.args.real_value:
                output = self.scaler.inverse_transform(output)
            loss = self.loss(output, label)
            loss.backward()
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % self.args.log_step == 0:
                self.logger.info(
                    'Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, self.train_per_epoch,
                        loss.item()))

        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, '
            'train time: {:.2f} s'.format(
                epoch, train_epoch_loss, time.time() - epoch_time))
        self.train_losses.append(train_epoch_loss)
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_test_model = None
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')

        self.logger.info("Start multi-scale training")
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            if train_epoch_loss > 1e6:
                self.logger.warning(
                    'Gradient explosion detected. Ending...')
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False

            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info(
                        "Validation performance didn't improve for {} "
                        "epochs. Training stops.".format(
                            self.args.early_stop_patience))
                    break

            if best_state:
                self.logger.info(
                    '*********************************'
                    'Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)

            # Also track best test loss
            test_dataloader = self.test_loader
            with torch.no_grad():
                self.model.eval()
                total_test_loss = 0
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data, target = self._batch_to_device(data, target)
                    label = target[..., :self.args.output_dim]
                    output = self.model(data)
                    if self.args.real_value:
                        output = self.scaler.inverse_transform(output)
                    loss = self.loss(output, label)
                    if not torch.isnan(loss):
                        total_test_loss += loss.item()
                test_epoch_loss = total_test_loss / len(test_dataloader)
            if test_epoch_loss < best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())

        # Save best model
        if not self.args.debug and best_model is not None:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            if best_test_model is not None:
                torch.save(best_test_model, self.best_test_path)
                self.logger.info(
                    "Saving best test model to " + self.best_test_path)

        # Test with best val model
        if best_model is not None:
            self.model.load_state_dict(best_model)
            self.logger.info("=== Best validation model results ===")
            self.test(self.model, self.args, self.test_loader,
                      self.scaler, self.logger, artifact_prefix="best_val")

        # Test with best test model
        if best_test_model is not None:
            self.model.load_state_dict(best_test_model)
            self.logger.info("=== Best test model results ===")
            self.test(self.model, self.args, self.test_loader,
                      self.scaler, self.logger, artifact_prefix="best_test")

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None,
             artifact_prefix=None):
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                label = target[..., :args.output_dim]
                output = model(data)
                if args.real_value:
                    output = scaler.inverse_transform(output)
                y_true.append(label.detach().cpu())
                y_pred.append(output.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        save_numpy_artifacts = os.environ.get("BASELINE_SAVE_NUMPY_ARTIFACTS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if save_numpy_artifacts:
            np.save('./{}_true.npy'.format(args.dataset), y_true_np)
            np.save('./{}_pred.npy'.format(args.dataset), y_pred_np)
        if getattr(args, "log_dir", None):
            output_dir = Path(args.log_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix = artifact_prefix or "test"
            if save_numpy_artifacts:
                np.save(output_dir / f"{prefix}_true.npy", y_true_np)
                np.save(output_dir / f"{prefix}_pred.npy", y_pred_np)
                if prefix == "best_val":
                    np.save(output_dir / "true.npy", y_true_np)
                    np.save(output_dir / "pred.npy", y_pred_np)
            meta = {
                "dataset": args.dataset,
                "model": args.model,
                "artifact_prefix": prefix,
                "selection": "validation" if prefix == "best_val" else prefix,
                "numpy_artifacts_saved": save_numpy_artifacts,
                "shape_true": list(y_true_np.shape),
                "shape_pred": list(y_pred_np.shape),
            }
            with (output_dir / f"{prefix}_prediction_meta.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, pcc = All_Metrics(
                y_pred[:, t, ...], y_true[:, t, ...],
                args.mae_thresh, args.mape_thresh)
            logger.info(
                "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, "
                "MAPE: {:.4f}".format(t + 1, mae, rmse, mape))
        mae, rmse, mape, _, pcc = All_Metrics(
            y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info(
            "Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, "
            "MAPE: {:.4f}".format(mae, rmse, mape))
