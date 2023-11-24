import numpy as np
import torch
from torch import nn

from hw_tts.base.base_model import BaseModel

from .decoder import Decoder
from .encoder import Encoder
from .length_regulator import LengthRegulator
from .model_config import FastSpeechConfig
from .sentiments_predictor import SentimentsPredictor


class FastSpeech2(BaseModel):
    def __init__(self):
        super().__init__()
        model_config = FastSpeechConfig()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, model_config.num_mels)

        self.pitch_emb = nn.Embedding(model_config.num_bins, model_config.encoder_dim)
        self.pitch_predictor = SentimentsPredictor(
            model_config.encoder_dim,
            predictor_filter_size=256,
            predictor_kernel_size=3,
            dropout=0.1,
        )
        pitch_space = torch.linspace(
            np.log(model_config.min_pitch + 1), 
            np.log(model_config.max_pitch + 2), 
            model_config.num_bins
        )
        self.register_buffer('pitch_space', pitch_space)

        self.energy_emb = nn.Embedding(model_config.num_bins, model_config.encoder_dim)
        self.energy_predictor = SentimentsPredictor(
            model_config.encoder_dim,
            predictor_filter_size=256,
            predictor_kernel_size=3,
            dropout=0.1,
        )
        energy_space = torch.linspace(
            np.log(model_config.min_energy + 1), 
            np.log(model_config.max_energy + 2), 
            model_config.num_bins
        )
        self.register_buffer('energy_space', energy_space)

    def get_pitch(self, x, pitch_target=None, beta=1.0):
        pitch_predictor_output = self.pitch_predictor(x)
        
        if pitch_target is not None:
            buckets = torch.bucketize(torch.log(pitch_target + 1), self.pitch_space)
        else:
            estimated_pitch = torch.exp(pitch_predictor_output) - 1
            estimated_pitch *= beta
            buckets = torch.bucketize(torch.log(estimated_pitch + 1), self.pitch_space)
        emb = self.pitch_emb(buckets)
        return emb, pitch_predictor_output

    def get_energy(self, x, energy_target=None, gamma=1.0):
        energy_predictor_output = self.energy_predictor(x)
        
        if energy_target is not None:
            buckets = torch.bucketize(torch.log(energy_target + 1), self.energy_space)
        else:
            estimated_energy = torch.exp(energy_predictor_output) - 1
            estimated_energy *= gamma
            buckets = torch.bucketize(torch.log(estimated_energy + 1), self.energy_space)
        emb = self.energy_emb(buckets)
        return emb, energy_predictor_output

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)


    def forward(
            self, 
            src_seq, 
            src_pos, 
            mel_pos=None,
            mel_max_length=None, 
            duration_target=None, 
            pitch_target=None, 
            energy_target=None,
            alpha=1.0, 
            beta=1.0, 
            gamma=1.0,
            **kwargs
        ):
        x, _ = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predictor_output = self.length_regulator(
                x, 
                alpha, 
                duration_target, 
                mel_max_length
            )

            pitch_emb, pitch_predictor_output = self.get_pitch(
                output, 
                pitch_target=pitch_target, 
                beta=beta
            )

            energy_emb, energy_predictor_output = self.get_energy(
                output, 
                energy_target=energy_target, 
                gamma=gamma
            )

            output = output + pitch_emb + energy_emb
            decoded = self.decoder(output, mel_pos)
            decoded_masked = self.mask_tensor(decoded, mel_pos, mel_max_length)
            mel_pred = self.mel_linear(decoded_masked)

            return {
                "mel_pred": mel_pred, 
                "duration_pred": duration_predictor_output,
                "pitch_pred": pitch_predictor_output,
                "energy_pred": energy_predictor_output
            }
        
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            pitch_emb, _ = self.get_pitch(output, beta=beta)
            energy_emb, _ = self.get_energy(output, gamma=gamma)
            output = output + pitch_emb + energy_emb
            decoded = self.decoder(output, mel_pos)
            mel_pred = self.mel_linear(decoded)
            return {"mel_pred": mel_pred}


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask