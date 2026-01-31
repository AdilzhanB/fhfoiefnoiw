"""
Speech Deep Learning Techniques Cheatsheet
Audio processing, ASR, TTS, speech recognition, and audio classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import scipy
from scipy import signal

# ============================================================================
# AUDIO PREPROCESSING & FEATURE EXTRACTION
# ============================================================================

def load_audio(file_path, sample_rate=16000):
    """Load audio file with specified sample rate"""
    waveform, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

# Mel Spectrogram extraction
def compute_mel_spectrogram(waveform, sample_rate=16000, n_fft=400, 
                           hop_length=160, n_mels=80, f_min=0, f_max=8000):
    """Compute mel spectrogram from waveform"""
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max
    )
    
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to log scale
    log_mel_spec = torch.log(mel_spec + 1e-9)
    
    return log_mel_spec

# MFCC (Mel-Frequency Cepstral Coefficients)
def compute_mfcc(waveform, sample_rate=16000, n_mfcc=40):
    """Compute MFCC features"""
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 80,
            'center': False
        }
    )
    
    mfcc = mfcc_transform(waveform)
    return mfcc

# Spectrogram augmentation
class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, time_masks=2, freq_width=15, time_width=20):
        super(SpecAugment, self).__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
    
    def forward(self, spec):
        # Frequency masking
        for _ in range(self.freq_masks):
            f = np.random.randint(0, self.freq_width)
            f0 = np.random.randint(0, spec.shape[1] - f)
            spec[:, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.time_masks):
            t = np.random.randint(0, self.time_width)
            t0 = np.random.randint(0, spec.shape[2] - t)
            spec[:, :, t0:t0+t] = 0
        
        return spec

# Pitch shifting
def pitch_shift(waveform, sample_rate, n_steps):
    """Shift pitch of audio"""
    waveform_np = waveform.numpy().squeeze()
    shifted = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)
    return torch.from_numpy(shifted).unsqueeze(0)

# Time stretching
def time_stretch(waveform, rate):
    """Stretch audio in time without changing pitch"""
    waveform_np = waveform.numpy().squeeze()
    stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
    return torch.from_numpy(stretched).unsqueeze(0)

# Noise addition
def add_noise(waveform, noise_level=0.005):
    """Add Gaussian noise to waveform"""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

# ============================================================================
# AUDIO CLASSIFICATION MODELS
# ============================================================================

class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(AudioCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# ResNet-based audio classifier
class AudioResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioResNet, self).__init__()
        # Use pretrained ResNet with modified first layer for spectrograms
        import torchvision.models as models
        resnet = models.resnet34(pretrained=True)
        
        # Modify first conv layer to accept 1-channel input (mel spectrogram)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ============================================================================
# RECURRENT MODELS FOR AUDIO
# ============================================================================

class AudioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super(AudioLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, freq, time)
        # Reshape for LSTM: (batch, time, freq)
        batch_size = x.size(0)
        x = x.squeeze(1).permute(0, 2, 1)  # (batch, time, freq)
        
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        
        return output

class AudioCRNN(nn.Module):
    """Convolutional Recurrent Neural Network for audio"""
    def __init__(self, num_classes, input_channels=1, hidden_dim=128):
        super(AudioCRNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout_conv = nn.Dropout(0.3)
        
        # RNN layers
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.dropout_rnn = nn.Dropout(0.5)
        
        # Fully connected
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)
        
        # Reshape for RNN: (batch, time, features)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, time, -1)
        
        # RNN
        x, (hidden, _) = self.lstm(x)
        x = self.dropout_rnn(x[:, -1, :])  # Take last time step
        
        # FC
        x = self.fc(x)
        
        return x

# ============================================================================
# TRANSFORMER-BASED AUDIO MODELS
# ============================================================================

class AudioTransformer(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(AudioTransformer, self).__init__()
        
        # Patch embedding for audio
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=(16, 1), stride=(16, 1))
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        # x: (batch, 1, freq, time)
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, d_model, freq', time)
        x = x.flatten(2).transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification from CLS token
        cls_output = x[:, 0]
        output = self.fc(cls_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# ============================================================================
# SPEECH RECOGNITION (ASR) - CTC
# ============================================================================

class DeepSpeech(nn.Module):
    """DeepSpeech-like architecture with CTC loss"""
    def __init__(self, num_classes, input_dim=80, hidden_dim=512, num_layers=5):
        super(DeepSpeech, self).__init__()
        
        # Initial convolution
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        
        # Calculate conv output size
        # After two convs with stride (2,2) and (2,1)
        conv_out_size = input_dim // 4
        rnn_input_size = 32 * conv_out_size
        
        # Recurrent layers
        self.rnns = nn.ModuleList([
            nn.LSTM(rnn_input_size if i == 0 else hidden_dim * 2, 
                   hidden_dim, 
                   bidirectional=True,
                   batch_first=True)
            for i in range(num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * 2) for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: (batch, 1, freq, time)
        x = self.conv(x)
        
        # Reshape for RNN: (batch, time, features)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, time, -1)
        
        # RNN layers
        for rnn, bn in zip(self.rnns, self.batch_norms):
            x, _ = rnn(x)
            x = bn(x.transpose(1, 2)).transpose(1, 2)
        
        # Output
        x = self.fc(x)
        
        return x  # (batch, time, num_classes)

# CTC Loss training
def train_ctc_model(model, train_loader, optimizer, device='cuda'):
    """Training loop for CTC-based ASR models"""
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    for spectrograms, labels, input_lengths, label_lengths in train_loader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(spectrograms)  # (batch, time, num_classes)
        output = output.log_softmax(2)
        output = output.transpose(0, 1)  # (time, batch, num_classes) for CTC
        
        # CTC loss
        loss = ctc_loss(output, labels, input_lengths, label_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return loss.item()

# CTC Greedy Decoder
def ctc_greedy_decode(output, blank_label=0):
    """Greedy decoding for CTC output"""
    # output: (time, num_classes)
    _, max_indices = output.max(dim=1)
    
    # Remove consecutive duplicates and blanks
    decoded = []
    prev_idx = None
    
    for idx in max_indices:
        idx = idx.item()
        if idx != blank_label and idx != prev_idx:
            decoded.append(idx)
        prev_idx = idx
    
    return decoded

# CTC Beam Search Decoder
def ctc_beam_search_decode(output, beam_width=10, blank_label=0):
    """Beam search decoding for CTC"""
    # Simplified beam search
    T, V = output.shape
    log_probs = torch.log_softmax(output, dim=1)
    
    # Initialize beam with empty sequence
    beams = [([], 0.0)]  # (sequence, score)
    
    for t in range(T):
        new_beams = []
        
        for seq, score in beams:
            for v in range(V):
                new_score = score + log_probs[t, v].item()
                
                if v == blank_label:
                    new_seq = seq
                elif len(seq) > 0 and seq[-1] == v:
                    new_seq = seq
                else:
                    new_seq = seq + [v]
                
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_width beams
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
    
    return beams[0][0]

# ============================================================================
# SPEECH RECOGNITION WITH TRANSFORMERS (Wav2Vec2, Whisper)
# ============================================================================

def transcribe_with_wav2vec2(audio_path, model_name='facebook/wav2vec2-base-960h'):
    """Transcribe audio using Wav2Vec2"""
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Process
    input_values = processor(waveform.squeeze().numpy(), 
                            sampling_rate=16000, 
                            return_tensors="pt").input_values
    
    # Transcribe
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def transcribe_with_whisper(audio_path, model_size='base'):
    """Transcribe audio using Whisper"""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    
    model_name = f'openai/whisper-{model_size}'
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Process
    input_features = processor(waveform.squeeze().numpy(), 
                               sampling_rate=16000, 
                               return_tensors="pt").input_features
    
    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

# ============================================================================
# TEXT-TO-SPEECH (TTS)
# ============================================================================

class Tacotron2Encoder(nn.Module):
    """Encoder for Tacotron 2"""
    def __init__(self, num_chars, embedding_dim=512):
        super(Tacotron2Encoder, self).__init__()
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2)
            for _ in range(3)
        ])
        
        self.bn = nn.ModuleList([nn.BatchNorm1d(embedding_dim) for _ in range(3)])
        self.lstm = nn.LSTM(embedding_dim, embedding_dim // 2, batch_first=True, bidirectional=True)
        
    def forward(self, text):
        x = self.embedding(text).transpose(1, 2)  # (batch, embed, seq_len)
        
        for conv, bn in zip(self.convs, self.bn):
            x = F.dropout(F.relu(bn(conv(x))), 0.5, self.training)
        
        x = x.transpose(1, 2)  # (batch, seq_len, embed)
        x, _ = self.lstm(x)
        
        return x

class TacotronAttention(nn.Module):
    """Location-sensitive attention for Tacotron"""
    def __init__(self, attention_dim, encoder_dim, decoder_dim):
        super(TacotronAttention, self).__init__()
        self.query_layer = nn.Linear(decoder_dim, attention_dim)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        
    def forward(self, query, memory):
        # query: (batch, decoder_dim)
        # memory: (batch, seq_len, encoder_dim)
        
        query = query.unsqueeze(1)  # (batch, 1, decoder_dim)
        processed_query = self.query_layer(query)  # (batch, 1, attention_dim)
        processed_memory = self.memory_layer(memory)  # (batch, seq_len, attention_dim)
        
        alignment = self.v(torch.tanh(processed_query + processed_memory))
        alignment = alignment.squeeze(-1)  # (batch, seq_len)
        
        attention_weights = F.softmax(alignment, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), memory)
        context = context.squeeze(1)  # (batch, encoder_dim)
        
        return context, attention_weights

# Simple TTS with pretrained models
def text_to_speech_pretrained(text, model_name='microsoft/speecht5_tts'):
    """Generate speech from text using pretrained TTS model"""
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset
    
    processor = SpeechT5Processor.from_pretrained(model_name)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    inputs = processor(text=text, return_tensors="pt")
    
    # Load speaker embeddings
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    return speech

# ============================================================================
# SPEAKER RECOGNITION / VERIFICATION
# ============================================================================

class SpeakerEmbeddingNetwork(nn.Module):
    """Extract speaker embeddings (x-vectors)"""
    def __init__(self, input_dim=80, embedding_dim=256):
        super(SpeakerEmbeddingNetwork, self).__init__()
        
        self.frame_layers = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        
        # Statistics pooling
        self.stat_pooling = StatisticsPooling()
        
        # Segment layers
        self.segment_layers = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        # x: (batch, freq, time)
        x = self.frame_layers(x)
        x = self.stat_pooling(x)
        x = self.segment_layers(x)
        return x

class StatisticsPooling(nn.Module):
    """Statistical pooling over time dimension"""
    def forward(self, x):
        # x: (batch, channels, time)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

# Contrastive loss for speaker verification
def contrastive_loss_speaker(embedding1, embedding2, label, margin=1.0):
    """Contrastive loss for speaker pairs"""
    distance = F.pairwise_distance(embedding1, embedding2)
    loss = (1 - label) * torch.pow(distance, 2) + \
           label * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

# ============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# ============================================================================

class VADModel(nn.Module):
    """Voice Activity Detection model"""
    def __init__(self, input_dim=40):
        super(VADModel, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 2)  # Speech or Non-speech
        
    def forward(self, x):
        # x: (batch, time, features)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

def detect_voice_activity(waveform, sample_rate=16000, frame_length=0.025, frame_stride=0.01):
    """Simple energy-based VAD"""
    frame_length_samples = int(frame_length * sample_rate)
    frame_stride_samples = int(frame_stride * sample_rate)
    
    # Compute frame energy
    frames = []
    for i in range(0, len(waveform) - frame_length_samples, frame_stride_samples):
        frame = waveform[i:i+frame_length_samples]
        energy = torch.sum(frame ** 2)
        frames.append(energy)
    
    energies = torch.tensor(frames)
    
    # Threshold-based detection
    threshold = torch.mean(energies) * 0.5
    voice_activity = energies > threshold
    
    return voice_activity

# ============================================================================
# AUDIO SOURCE SEPARATION
# ============================================================================

class UNetSeparation(nn.Module):
    """U-Net for audio source separation"""
    def __init__(self, num_sources=2):
        super(UNetSeparation, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decoder
        self.dec4 = self.conv_block(384, 128)
        self.dec3 = self.conv_block(192, 64)
        self.dec2 = self.conv_block(96, 32)
        self.dec1 = self.conv_block(48, 16)
        
        self.out = nn.Conv2d(16, num_sources, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.out(dec1)
        return out

# ============================================================================
# AUDIO GENERATION
# ============================================================================

class WaveGAN(nn.Module):
    """WaveGAN Generator for raw audio generation"""
    def __init__(self, latent_dim=100, audio_length=16384):
        super(WaveGAN, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 16)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=25, stride=4, padding=11),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, kernel_size=25, stride=4, padding=11),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=25, stride=4, padding=11),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=25, stride=4, padding=11),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=25, stride=4, padding=11),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.fc(z).view(-1, 256, 16)
        x = self.deconv(x)
        return x

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Levenshtein distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
    
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer

def calculate_pesq(reference_audio, degraded_audio, sample_rate=16000):
    """Calculate PESQ (Perceptual Evaluation of Speech Quality)"""
    from pesq import pesq
    
    score = pesq(sample_rate, reference_audio, degraded_audio, 'wb')
    return score

def calculate_stoi(reference_audio, degraded_audio, sample_rate=16000):
    """Calculate STOI (Short-Time Objective Intelligibility)"""
    from pystoi import stoi
    
    score = stoi(reference_audio, degraded_audio, sample_rate, extended=False)
    return score
import torch
import numpy as np
import librosa
from typing import Dict, List, Union, Literal
from tqdm.auto import tqdm
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2Processor, 
    WhisperModel, 
    WhisperProcessor
)

class AudioEmbeddingExtractor:
    def __init__(self, 
                 mode: Literal['wav', 'whisper', 'fusion'] = 'wav', 
                 device: str = None):
        self.mode = mode
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Initializing AudioEmbeddingExtractor on {self.device} in '{self.mode}' mode...")
        self._load_models()

    def _load_models(self):
        if self.mode in ['wav', 'fusion']:
            self.w2v_name = "facebook/wav2vec2-base-960h"
            self.w2v_processor = Wav2Vec2Processor.from_pretrained(self.w2v_name)
            self.w2v_model = Wav2Vec2Model.from_pretrained(self.w2v_name).to(self.device)
            self.w2v_model.eval()

        if self.mode in ['whisper', 'fusion']:
            self.whisper_name = "openai/whisper-base"
            self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_name)
            self.whisper_model = WhisperModel.from_pretrained(self.whisper_name).to(self.device)
            self.whisper_model.config.forced_decoder_ids = None 
            self.whisper_model.eval()

    def _load_audio_batch(self, paths: List[str], min_samples: int = 4000) -> List[np.ndarray]:
        """Loads audio and pads extremely short files to prevent Conv1D errors."""
        batch_audio = []
        for path in paths:
            try:
                audio, _ = librosa.load(path, sr=16000)
                # Pad if shorter than 0.25s
                if len(audio) < min_samples:
                    padding = min_samples - len(audio)
                    audio = np.pad(audio, (0, padding), 'constant')
                batch_audio.append(audio)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                batch_audio.append(np.zeros(min_samples))
        return batch_audio

    def _get_wav2vec_embeddings(self, raw_audio: List[np.ndarray], max_length_samples: int) -> torch.Tensor:
        """
        Wav2Vec2 supports variable lengths. We use the user-defined max_length.
        """
        inputs = self.w2v_processor(
            raw_audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True,        # Pad to longest in batch
            truncation=True,     # Cut if longer than max_length
            max_length=max_length_samples
        ).to(self.device)

        with torch.no_grad():
            outputs = self.w2v_model(**inputs)
            # Mean pool over the sequence length
            pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return pooled

    def _get_whisper_embeddings(self, raw_audio: List[np.ndarray]) -> torch.Tensor:
        """
        Whisper STRICTLY requires 30 seconds of input (3000 mel frames).
        We force padding='max_length' which uses the model's config (30s).
        """
        inputs = self.whisper_processor(
            raw_audio, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding="max_length", # <--- FIX: Forces 30s padding regardless of input length
            truncation=True       # Truncate if > 30s
        ).to(self.device)
        
        input_features = inputs.input_features

        with torch.no_grad():
            outputs = self.whisper_model.encoder(input_features)
            # outputs.last_hidden_state is always (Batch, 1500, 512) for base model
            # Mean pool
            pooled = torch.mean(outputs.last_hidden_state, dim=1)
        return pooled

    def extract_embeddings(self, 
                           audio_paths: List[str], 
                           batch_size: int = 16,
                           max_length_seconds: int = 5) -> torch.Tensor:
        """
        Args:
            audio_paths: List of file paths.
            batch_size: Batch size.
            max_length_seconds: Applied to Wav2Vec2 ONLY. Whisper is always padded to 30s.
        """
        all_embeddings = []
        total_files = len(audio_paths)
        
        # Max samples for Wav2Vec2
        w2v_max_samples = max_length_seconds * 16000

        print(f"Processing {total_files} files | Batch Size: {batch_size}")
        if self.mode == 'wav' or self.mode == 'fusion':
            print(f"Wav2Vec2 max length set to {max_length_seconds}s")
        if self.mode == 'whisper' or self.mode == 'fusion':
            print(f"Whisper inputs will be auto-padded to 30s (Model Requirement)")

        for i in tqdm(range(0, total_files, batch_size), desc="Extracting"):
            batch_paths = audio_paths[i : i + batch_size]
            raw_audio = self._load_audio_batch(batch_paths)
            
            batch_emb = None

            # --- MODE HANDLING ---
            if self.mode == 'wav':
                batch_emb = self._get_wav2vec_embeddings(raw_audio, w2v_max_samples)

            elif self.mode == 'whisper':
                # Note: We do NOT pass max_length here. Whisper handles it.
                batch_emb = self._get_whisper_embeddings(raw_audio)

            elif self.mode == 'fusion':
                w2v_emb = self._get_wav2vec_embeddings(raw_audio, w2v_max_samples)
                whisper_emb = self._get_whisper_embeddings(raw_audio)
                # Concatenate (Batch, 768) + (Batch, 512) -> (Batch, 1280)
                batch_emb = torch.cat((w2v_emb, whisper_emb), dim=1)
            
            all_embeddings.append(batch_emb.cpu())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not all_embeddings:
            return torch.tensor([])
            
        final_tensor = torch.cat(all_embeddings, dim=0)
        return final_tensor

# --- Usage Example ---
if __name__ == "__main__":
    # Your paths
    # paths = train.sampleID.values.tolist()
    # paths = ["/kaggle/input/.../" + p + ".wav" for p in paths]
    
    # Dummy setup for verification
    import soundfile as sf
    dummy_name = "test_short.wav"
    sf.write(dummy_name, np.random.uniform(-1, 1, 16000*5), 16000) # 5 seconds
    paths = [dummy_name] * 4

    # Run Fusion
    extractor = AudioEmbeddingExtractor(mode='fusion')
    
    # Even though we say 5s, Whisper will internally pad to 30s, 
    # Wav2Vec2 will stick to 5s. The code handles this difference.
    embeddings = extractor.extract_embeddings(paths, batch_size=2, max_length_seconds=5)
    
    print(f"Final Embeddings Shape: {embeddings.shape}")
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    WhisperFeatureExtractor, 
    WhisperModel, 
    TrainingArguments, 
    Trainer
)
from torch import nn
import evaluate

# --- CONFIGURATION ---
MODEL_CHECKPOINT = "openai/whisper-tiny"
dataset_name = "superb"
dataset_config = "ks"  # Keyword Spotting
batch_size = 32
num_labels = 12  # The dataset has 12 keywords (yes, no, up, down, etc.)
MAX_DURATION = 1.0 # Seconds (short for keyword spotting)

# 1. LOAD DATASET
# ---------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset(dataset_name, dataset_config)

# Label mapping (Label ID -> Name)
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# 2. PREPROCESSING
# ---------------------------------------------------------
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        return_tensors="pt",
        truncation=True,
        padding="max_length", # Pad to 30s usually, but for speed we rely on extractor defaults
        max_length=int(MAX_DURATION * 16000) 
    )
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["audio", "file"])

# 3. DEFINE CUSTOM MODEL
# ---------------------------------------------------------
class WhisperForAudioClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Load Base Whisper (Encoder + Decoder)
        self.whisper = WhisperModel.from_pretrained(MODEL_CHECKPOINT)
        
        # We only need the Encoder for classification
        self.encoder = self.whisper.encoder
        
        # FREEZE the encoder to prevent destroying pretrained features
        # (Optional: Unfreeze later for better performance)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.whisper.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_features, labels=None):
        # input_features shape: [Batch, 80, 3000]
        outputs = self.encoder(input_features)
        
        # Mean pooling: Average over the time dimension
        # outputs.last_hidden_state shape: [Batch, Time, Dim]
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.long())
            
        return (loss, logits) if loss is not None else logits

model = WhisperForAudioClassification(num_labels=num_labels)

# 4. METRICS
# ---------------------------------------------------------
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

# 5. TRAINING
# ---------------------------------------------------------
# We need a custom data collator to handle the specific input format
def data_collator(features):
    input_features = torch.stack([torch.tensor(f["input_features"][0]) for f in features])
    labels = torch.tensor([f["label"] for f in features])
    return {"input_features": input_features, "labels": labels}

training_args = TrainingArguments(
    output_dir="./whisper_classifier",
    per_device_train_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=1e-4,
    remove_unused_columns=False, # Important for custom models
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate

# --- CONFIGURATION ---
MODEL_ID = "openai/whisper-tiny"
OUT_DIR = "./whisper-finetuned-stt"
# We use a tiny slice of Common Voice 11 (Lithuanian used as example of low-resource lang)
# You can change 'lt' to 'en', 'hi', etc.
DATASET_ID = "mozilla-foundation/common_voice_11_0" 
LANGUAGE = "lt" 
TASK = "transcribe"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. LOAD DATASET & PROCESSOR
# ------------------------------------------------------------------
print("Loading dataset and processor...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)

# Load streaming=True to avoid downloading 100GB. We take a small sample.
common_voice = load_dataset(DATASET_ID, LANGUAGE, split="train", streaming=True)
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Take 200 samples for training, 50 for validation
train_dataset = list(common_voice.take(200))
eval_dataset = list(common_voice.skip(200).take(50))

# Convert list back to HuggingFace Dataset object for easier mapping
from datasets import Dataset
train_dataset = Dataset.from_list(train_dataset)
eval_dataset = Dataset.from_list(eval_dataset)

# 2. PREPROCESSING
# ------------------------------------------------------------------
def prepare_dataset(batch):
    # Process Audio
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Process Text
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("Preprocessing data...")
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)

# 3. DATA COLLATOR (Handles Padding dynamically)
# ------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Treat inputs and labels differently for padding
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous step, cut it (whisper adds it automatically)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 4. METRICS (Word Error Rate)
# ------------------------------------------------------------------
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 5. MODEL & TRAINING
# ------------------------------------------------------------------
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=200, # Short run for demo
    gradient_checkpointing=True,
    fp16=True if torch.cuda.is_available() else False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["none"], # Disable wandb for this demo
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

print(f"Training finished. Model saved to {OUT_DIR}")
import torch
from datasets import load_dataset, Audio
from transformers import (
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# --- CONFIGURATION ---
MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan" # Needed to listen to results
OUT_DIR = "./speecht5-finetuned-tts"
dataset_id = "lj_speech"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. LOAD DATASET & PROCESSOR
# ------------------------------------------------------------------
print("Loading Processor and Model...")
processor = SpeechT5Processor.from_pretrained(MODEL_ID)

print("Loading LJ Speech dataset (taking small subset)...")
# Taking 200 examples for speed
dataset = load_dataset(dataset_id, split="train[:200]") 
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 2. CREATE SPEAKER EMBEDDINGS (X-Vectors)
# ------------------------------------------------------------------
# SpeechT5 is a multi-speaker model. Even for single speaker fine-tuning, 
# we need to pass a speaker embedding. We will use a default one.
# Normally you extract this from the audio using 'speechbrain', but we will
# fetch a pre-calculated one to keep this script simple.
from datasets import load_dataset as ld_xvector
embeddings_ds = ld_xvector("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_ds[0]["xvector"]).unsqueeze(0)

# 3. PREPROCESSING
# ------------------------------------------------------------------
def prepare_dataset(example):
    # 1. Process Text
    example["input_ids"] = processor(text=example["text"]).input_ids
    
    # 2. Process Audio (Target)
    audio = example["audio"]
    # The model expects Mel Spectrograms as targets, not raw waveforms
    example["labels"] = processor(
        audio=audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).audio_values
    
    return example

print("Preprocessing dataset...")
dataset = dataset.map(prepare_dataset, remove_columns=["audio", "text", "normalized_text", "id"])
# Train/Test Split
dataset = dataset.train_test_split(test_size=0.1)

# 4. DATA COLLATOR
# ------------------------------------------------------------------
@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        
        # Pad Text Inputs
        batch = self.processor.tokenizer.pad(input_ids, return_tensors="pt")
        
        # Pad Audio Targets (Spectrograms)
        labels_batch = self.processor.feature_extractor.pad(label_features, return_tensors="pt")
        
        # Reshape labels to correct format
        labels = labels_batch["input_values"]
        labels_mask = labels_batch.attention_mask
        
        # Replace padding with -100 for loss calculation
        labels = labels.masked_fill(labels_mask.unsqueeze(-1).ne(1), -100)
        
        batch["labels"] = labels
        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

# 5. LOAD MODEL
# ------------------------------------------------------------------
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)

# Disable "use_cache" during training
model.config.use_cache = False 

# 6. TRAINING
# ------------------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8, # reduced for GPU mem
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=300,
    gradient_checkpointing=True,
    fp16=True if torch.cuda.is_available() else False,
    evaluation_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["none"],
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

print("Starting TTS training...")
trainer.train()

# 7. INFERENCE TEST (Generate Audio)
# ------------------------------------------------------------------
print("Generating test audio...")
model.eval()
model.to(device)

text = "This is a test of the fine tuned text to speech model."
inputs = processor(text=text, return_tensors="pt").to(device)

# Load Vocoder (converts spectrogram -> sound)
vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID).to(device)

with torch.no_grad():
    spectrogram = model.generate_speech(
        inputs["input_ids"], 
        speaker_embeddings=speaker_embedding.to(device)
    )
    # Convert spectrogram to waveform using Vocoder
    speech = vocoder(spectrogram)

# Save to file
import soundfile as sf
sf.write("tts_result.wav", speech.cpu().numpy(), samplerate=16000)
print("Saved output to tts_result.wav")
