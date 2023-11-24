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
        src_pos = self.get_src_pos(torch.from_numpy(tokens)).to(device='cuda')
        tokens = torch.from_numpy(tokens).unsqueeze(0).to(device='cuda')
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                src_seq=tokens,
                src_pos=src_pos,
                **generation_params
            )
        mel = output['mel_pred'].transpose(1, 2)
        audio = self.converter.mel_to_wave(mel)
        return audio
    
    def get_src_pos(self, tokens):
        texts = [tokens]
        length_text = []
        for text in texts:
            length_text.append(text.size(0))
        length_text = np.array(length_text)

        src_pos = list()
        max_len = int(max(length_text))
        for length_src_row in length_text:
            src_pos.append(np.pad(
                [i + 1 for i in range(int(length_src_row))],
                (0, max_len-int(length_src_row)),
                'constant'
            ))
        src_pos = torch.from_numpy(np.array(src_pos))
        return src_pos