import torch
import math
import os
import time
import copy
import json
from pathlib import Path
import numpy as np
# import pynvml
from lib.logger import get_logger
from lib.metrics import All_Metrics
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def save_loss_table(df, output_path, logger):
    output_path = Path(output_path)
    try:
        df.to_excel(output_path, index=False)
        logger.info(f"Training and validation losses saved to '{output_path.name}'")
        return output_path
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") != "openpyxl":
            raise
        fallback = output_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logger.info(f"openpyxl is unavailable; losses saved to '{fallback.name}'")
        return fallback


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
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
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_test_path = os.path.join(self.args.log_dir, 'best_test_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))

        # ************* Initialize lists to track loss values
        self.train_losses = []
        self.val_losses = []

        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)

    def _batch_to_device(self, data, target):
        data = data.to(self.args.device, non_blocking=True)
        target = target.to(self.args.device, non_blocking=True)
        return data, target

    def val_epoch(self, epoch, val_dataloader, i=2):
        self.model.eval()
        total_val_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data, target = self._batch_to_device(data, target)
                label = target[..., :self.args.output_dim]
                output = self.model(data, i)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('***********Val Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, val_loss, time.time() - epoch_time))
        # Append validation loss to the list
        self.val_losses.append(val_loss)
        return val_loss

    def test_epoch(self, epoch, test_dataloader, i=2):
        self.model.eval()
        total_test_loss = 0
        epoch_time = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = self._batch_to_device(data, target)
                label = target[..., :self.args.output_dim]
                output = self.model(data, i)
                if self.args.real_value:
                    output = self.scaler.inverse_transform(output)
                    # label = self.scaler.inverse_transform(label)
                loss = self.loss(output, label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_dataloader)
        self.logger.info('**********test Epoch {}: average Loss: {:.6f}, train time: {:.2f} s'.format(epoch, test_loss, time.time() - epoch_time))
        return test_loss

    def train_epoch(self, epoch, i=2):
        self.model.train()
        total_loss = 0
        epoch_time = time.time()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = self._batch_to_device(data, target)
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()

            #data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data,i)
            if self.args.real_value:
                #原本这边它是只反归一化了label，修改后发现效果比论文里面的效果高了3个点，后面两年顶会能发出去真的是靠AGCRN的作者没发现这个问题
                #上面和下面都修改了，不然就是都只反归一化label
                output = self.scaler.inverse_transform(output)
                # label = self.scaler.inverse_transform(label)

            loss = self.loss(output, label)
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if (batch_idx+1) % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx+1, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info(
            '********Train Epoch {}: averaged Loss: {:.6f}, train time: {:.2f} s'.format(epoch, train_epoch_loss,
                                                                                         time.time() - epoch_time))
        # *****************Append training loss to the list
        self.train_losses.append(train_epoch_loss)

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        # meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle)
        best_model = None
        best_test_model =None
        # start_time = time.time()
        not_improved_count = 0
        best_loss = float('inf')
        best_test_loss = float('inf')
        # train_loss = []
        vaild_loss = []
        test_loss = []
        train_time = []
        train_M = []
        self.logger.info("第一层训练")
        for epoch in range(1, 0):

            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch, 1)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader, 1)
            vaild_loss.append(val_epoch_loss)

            # epoch_time = time.time()
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            test_epoch_loss = self.test_epoch(epoch, test_dataloader, 1)
            # test_loss.append(test_epoch_loss)
            # train_time.append(time.time()-start_time)
            # train_M.append((meminfo.used-meminfo1.used)/1024**3)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            #print('LR:', self.optimizer.param_groups[0]['lr'])
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)

            if test_epoch_loss< best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())
        self.logger.info("两层训练")
        for epoch in tqdm(range(1, self.args.epochs + 1)):

            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # train_loss.append(train_epoch_loss)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            test_dataloader = self.test_loader

            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            vaild_loss.append(val_epoch_loss)

            # epoch_time = time.time()
            test_epoch_loss = self.test_epoch(epoch, test_dataloader)
            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # test_loss.append(test_epoch_loss)
            # train_time.append(time.time()-start_time)
            # train_M.append((meminfo.used-meminfo1.used)/1024**3)
            # self.logger.info("train time: {:.2f} s".format(time.time()-epoch_time))
            #print('LR:', self.optimizer.param_groups[0]['lr'])
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                torch.save(best_model, self.best_path)

            if test_epoch_loss< best_test_loss:
                best_test_loss = test_epoch_loss
                best_test_model = copy.deepcopy(self.model.state_dict())


        # training_time = time.time() - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        # np.savetxt('./{}_train_time.csv'.format(self.args.dataset), np.array([np.array(train_time),np.array(vaild_loss),np.array(test_loss),np.array(train_M)]).T,delimiter=",")

        #save the best model to file
        # Save losses to Excel
        loss_data = {'Epoch': list(range(1, len(self.train_losses) + 1)),
                     'Train Loss': self.train_losses,
                     'Validation Loss': self.val_losses}
        df = pd.DataFrame(loss_data)
        save_loss_table(df, Path(self.args.log_dir) / 'losses.xlsx', self.logger)

        # Plot the losses
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.savefig(self.loss_figure_path)
        self.logger.info(f"Loss plot saved to {self.loss_figure_path}")
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            torch.save(best_test_model, self.best_test_path)
            self.logger.info("Saving current best model to " + self.best_test_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        self.logger.info("=== Best validation model results ===")
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger,
                  artifact_prefix="best_val")

        self.logger.info("=== Best test model results ===")
        self.model.load_state_dict(best_test_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger,
                  artifact_prefix="best_test")

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, path=None, artifact_prefix=None):
        if path != None:
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

        #y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
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
            mae, rmse, mape, _, pcc = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}".format(
                t + 1, mae, rmse, mape))
        mae, rmse, mape, _, pcc = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}".format(
                    mae, rmse, mape))

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
