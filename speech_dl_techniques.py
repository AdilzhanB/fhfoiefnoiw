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
