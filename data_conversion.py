import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
import os

class SpeechCommandsComplexSTFT(SPEECHCOMMANDS):
    """
    Wrapper for Google Speech Commands to match REVERB-FL Signal Modeling.
    """
    
    # subset defaults to None, but you will pass "training" or "testing"
    def __init__(self, root="./", subset=None, download=True):
        # Creates a folder for the path if it doesnt already exist
        os.makedirs(root, exist_ok=True)
        
        # 1. PyTorch handles the .txt files automatically based on 'subset'
        super().__init__(root=root, download=download, subset=subset)
        
        # 2. Define the 10-class subset
        self.target_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        self.label_to_index = {label: i for i, label in enumerate(self.target_labels)}
        
        # 3. Filter PyTorch's file list to only include our 10 words
        self._walker = [
            fileid for fileid in self._walker 
            if fileid.split(os.sep)[-2] in self.target_labels
        ]
        
        # 4. STFT Parameters
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.window = torch.hann_window(self.win_length)

    def __getitem__(self, n):
        """
        Currently, the data is not being normalized at this step to 0 and 1,
        so the poison attack functions need to do dynamic clipping instead of
        setting it from 0 to 1.
        """

        # A. Load Raw Waveform
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        
        # B. Pad/Truncate to 1 second (16000 samples)
        target_length = 16000
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_length]
            
        # C. Compute Complex STFT
        stft_complex = torch.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True
        )
        
        # D. View as Real/Imag Components: (1, F, T, 2)
        stft_real_imag = torch.view_as_real(stft_complex)
        
        # E. Squeeze out the extra dimension: (F, T, 2)
        x_vector = stft_real_imag.squeeze(0)

        x_vector = x_vector.permute(2, 0, 1)
        
        # G. Get Label Index
        label_idx = self.label_to_index[label]
        
        return x_vector, label_idx
