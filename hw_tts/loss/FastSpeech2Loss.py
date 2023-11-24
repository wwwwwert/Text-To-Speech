import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR


class FastSpeech2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_loss = torch.nn.MSELoss()
        self.duration_loss = torch.nn.MSELoss()
        self.energy_loss = torch.nn.MSELoss()
        self.pitch_loss = torch.nn.MSELoss()

    def forward(
            self, 
            mel_pred, 
            duration_pred, 
            pitch_pred, 
            energy_pred, 
            mel_target, 
            duration_target,
            energy_target, 
            pitch_target, 
            **kwargs
        ):
        mel_loss = self.mel_loss(mel_pred, mel_target)

        duration_loss = self.duration_loss(
            duration_pred,
            torch.log((duration_target + 1).float())
        )

        energy_loss = self.energy_loss(
            energy_pred,
            torch.log(energy_target + 1)
        )

        pitch_loss = self.pitch_loss(
            pitch_pred,
            torch.log(pitch_target + 1)
        )

        loss = mel_loss + duration_loss + energy_loss + pitch_loss

        return {
            "loss": loss,
            "mel_loss": mel_loss, 
            "duration_loss": duration_loss, 
            "energy_loss": energy_loss, 
            "pitch_loss": pitch_loss
        }
    
