import random
from pathlib import Path
from random import shuffle

import pandas as pd
import PIL
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.base.base_text_encoder import BaseTextEncoder
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import MetricTracker, inf_loop
from waveglow.converter import MelToWave

import matplotlib.pyplot as plt
import numpy as np

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 2400

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "energy_loss", "pitch_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "energy_loss", "pitch_loss", *[m.name for m in self.metrics], writer=self.writer
        )

        self.sample_rate = config["preprocessing"]["sr"]
        self.accum_iter = 4
        self.converter = MelToWave()
        self.converter.wave_glow.to(torch.device('cpu'))

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        to_device = [
            "src_seq",
            "mel_target",
            "length_target",
            "pitch_target",
            "energy_target",
            "mel_pos",
            "src_pos",
            # "mel_max_len"
        ]
        for tensor_for_gpu in to_device:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for list_batch_idx, list_of_batches in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            stop = False
            for batch_idx, batch in enumerate(list_of_batches):
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                real_batch_idx = batch_idx + list_batch_idx * 24
                if real_batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + real_batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(real_batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_predictions(**batch)
                    # self._log_spectrogram(batch["spectrogram"])
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if real_batch_idx + 1 >= self.len_epoch:
                    stop = True
                    break
            if stop:
                break
            torch.cuda.empty_cache()
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_idx:int=1):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)
        
        loss = self.criterion(**batch)
        batch.update(loss)

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        n_val = 100
        i = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=min(n_val, len(dataloader)),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
                i += 1
                if i >= n_val:
                    break

            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            mel_pred,
            mel_target,
            text,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        mel_pred = mel_pred.cpu()
        mel_target = mel_target.cpu()

        # print('len text:', len(text))
        # print('mel_pred:', mel_pred.shape)
        # print('mel_target:', mel_target.shape)

        tuples = list(zip(mel_pred, mel_target, text))
        shuffle(tuples)
        rows = []

        examples_to_log = 10
        self.converter.wave_glow.to(torch.device('cuda'))

        for mel_pred_item, mel_target_item, text_item in tuples[:examples_to_log]:
            with torch.no_grad():
                audio_pred_item = self.converter.mel_to_wave(mel_pred_item.transpose(0, 1).unsqueeze(0).to(torch.device('cuda'))).squeeze()
                audio_target_item = self.converter.mel_to_wave(mel_target_item.transpose(0, 1).unsqueeze(0).to(torch.device('cuda'))).squeeze()
            
            mel_pred_img = mel_pred_item.transpose(0, 1).detach().numpy()
            mel_target_img = mel_target_item.transpose(0, 1).detach().numpy() + 1
            
            plt.cla()
            plt.imshow(mel_target_img)
            plt.savefig('mel.png')
            row = [
                text_item,
                self.writer.wandb.Image(mel_pred_img),
                self.writer.wandb.Image(mel_target_img),
                self.writer.wandb.Audio(audio_pred_item, sample_rate=self.sample_rate),
                self.writer.wandb.Audio(audio_target_item, sample_rate=self.sample_rate),
            ]
            rows.append(row)
        self.converter.wave_glow.to(torch.device('cpu'))
        predictions = pd.DataFrame(rows, columns=[
            'text',
            'pred MEL',
            'target MEL',
            'pred audio',
            'target audio',
        ])
        self.writer.add_table("predictions", predictions)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
