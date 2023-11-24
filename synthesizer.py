import numpy as np
import torch

from waveglow.converter import MelToWave
from waveglow.text import text_to_sequence


class Synthesizer():
    def __init__(self, model):
        self.model = model
        self.converter = MelToWave()
        self.text_cleaners = ['english_cleaners']

    def synthesize(self, text, generation_params):
        tokens = np.array(
            text_to_sequence(text, self.text_cleaners)
        )
        tokens = torch.from_numpy(tokens).unsqueeze(0)
        src_pos = torch.tensor([[tokens.shape[1]]])
        self.model.eval()
        with torch.no_grad():
            mel = self.model(
                src_seq=tokens,
                src_pos=src_pos,
                **generation_params
            )
        audio = self.converter(mel)
        return audio