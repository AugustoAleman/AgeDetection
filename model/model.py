import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import torch.nn as nn

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

class AudioProcessor:
    def __init__(self, model_name='audeering/wav2vec2-large-robust-24-ft-age-gender'):
        self.device = 'cpu'
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AgeGenderModel.from_pretrained(model_name).to(self.device)

    def preprocess_audio(self, waveform, sample_rate):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        return waveform.numpy()

    def predict(self, signal, sampling_rate, embeddings=False):
        inputs = self.processor(signal, sampling_rate=sampling_rate)['input_values'][0]
        inputs = torch.tensor(inputs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            if embeddings:
                return outputs[0].cpu().numpy()
            else:
                age = outputs[1].cpu().numpy()[0, 0]
                gender_probs = outputs[2].cpu().numpy()[0]
                return {"age": float(age), "gender_probs": gender_probs.tolist()}
