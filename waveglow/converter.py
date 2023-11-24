import torch
from scipy.io.wavfile import write

MAX_WAV_VALUE = 32768.0


class MelToWave:
    def __init__(self, waveglow_path: str='waveglow/waveglow_256channels_ljs_v2.pt') -> None:
        self.wave_glow = torch.load(waveglow_path)['model']
        self.wave_glow = self.wave_glow.remove_weightnorm(self.wave_glow)
        for m in self.wave_glow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

    def mel_to_wave(self, mel: torch.Tensor, audio_path: str=None, sigma=1.0, sampling_rate=22050):
        self.wave_glow.eval()
        
        with torch.no_grad():
            audio = self.wave_glow.infer(mel, sigma=sigma)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        if audio_path is not None:
            write(audio_path, sampling_rate, audio)
        return audio