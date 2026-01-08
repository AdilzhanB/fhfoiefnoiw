"""
Time Series Analysis and Forecasting Techniques Cheatsheet
From classical methods (ARIMA) to deep learning (LSTM, Transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# TIME SERIES PREPROCESSING
# ============================================================================

def check_stationarity(timeseries, significance_level=0.05):
    """Check if time series is stationary using ADF test"""
    result = adfuller(timeseries, autolag='AIC')
    
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    is_stationary = result[1] < significance_level
    print(f'\nStationary: {is_stationary}')
    
    return is_stationary

def make_stationary(timeseries, method='difference'):
    """Make time series stationary"""
    if method == 'difference':
        # First order differencing
        return timeseries.diff().dropna()
    elif method == 'log':
        # Log transformation
        return np.log(timeseries)
    elif method == 'log_difference':
        # Log + differencing
        return np.log(timeseries).diff().dropna()
    elif method == 'seasonal_difference':
        # Seasonal differencing
        return timeseries.diff(12).dropna()  # 12 for monthly data

def decompose_timeseries(timeseries, period, model='additive'):
    """Decompose time series into trend, seasonal, and residual"""
    decomposition = seasonal_decompose(timeseries, model=model, period=period)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': decomposition.observed
    }

def create_sequences(data, seq_length, forecast_horizon=1):
    """Create sequences for supervised learning"""
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    
    return np.array(X), np.array(y)

def scale_timeseries(train_data, test_data=None, method='standard'):
    """Scale time series data"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
    
    if test_data is not None:
        test_scaled = scaler.transform(test_data.reshape(-1, 1))
        return train_scaled, test_scaled, scaler
    
    return train_scaled, scaler

# ============================================================================
# CLASSICAL TIME SERIES MODELS
# ============================================================================

def fit_arima(timeseries, order=(1, 1, 1)):
    """Fit ARIMA model"""
    model = ARIMA(timeseries, order=order)
    fitted_model = model.fit()
    
    print(fitted_model.summary())
    return fitted_model

def fit_sarima(timeseries, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Fit SARIMA model for seasonal data"""
    model = SARIMAX(timeseries, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    
    print(fitted_model.summary())
    return fitted_model

def auto_arima_selection(timeseries, max_p=5, max_d=2, max_q=5):
    """Grid search for best ARIMA parameters"""
    import itertools
    
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)
    
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(timeseries, order=(p, d, q))
            fitted = model.fit()
            
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_model = fitted
        except:
            continue
    
    print(f'Best ARIMA order: {best_order} with AIC: {best_aic}')
    return best_model, best_order

def exponential_smoothing(timeseries, seasonal_periods=12, trend='add', seasonal='add'):
    """Exponential Smoothing (Holt-Winters)"""
    model = ExponentialSmoothing(
        timeseries,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal
    )
    fitted_model = model.fit()
    
    return fitted_model

# ============================================================================
# LSTM FOR TIME SERIES
# ============================================================================

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(VanillaLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use last time step
        
        return out

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out

class StackedLSTM(nn.Module):
    """Stacked LSTM with attention"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(StackedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output

# ============================================================================
# GRU FOR TIME SERIES
# ============================================================================

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# ============================================================================
# ENCODER-DECODER LSTM (SEQ2SEQ)
# ============================================================================

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(EncoderLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(DecoderLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon):
        super(Seq2SeqLSTM, self).__init__()
        
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(input_size, hidden_size, num_layers, output_size)
        self.forecast_horizon = forecast_horizon
        
    def forward(self, x):
        # Encode
        hidden, cell = self.encoder(x)
        
        # Decode
        decoder_input = x[:, -1:, :]  # Last time step
        predictions = []
        
        for _ in range(self.forecast_horizon):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions.append(prediction)
            decoder_input = prediction
        
        predictions = torch.cat(predictions, dim=1)
        return predictions

# ============================================================================
# ATTENTION-BASED MODELS
# ============================================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWithAttention, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

# ============================================================================
# TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, output_size=1):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]  # Take last time step
        return self.fc(y)

# ============================================================================
# TRANSFORMER FOR TIME SERIES
# ============================================================================

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # Use last time step
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ============================================================================
# N-BEATS (Neural Basis Expansion Analysis)
# ============================================================================

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super(NBeatsBlock, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(input_size if i == 0 else layer_size, layer_size) 
                                     for i in range(layers)])
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        theta = self.basis_parameters(x)
        backcast, forecast = self.basis_function(theta)
        
        return backcast, forecast

class NBeats(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=3, num_layers=4, layer_size=256):
        super(NBeats, self).__init__()
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, output_size, 
                       lambda x: (x[:, :input_size], x[:, input_size:]),
                       num_layers, layer_size)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        forecast = torch.zeros(x.size(0), x.size(1)).to(x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(x)
            x = x - backcast
            forecast = forecast + block_forecast
        
        return forecast

# ============================================================================
# PROPHET-STYLE DECOMPOSITION
# ============================================================================

def prophet_forecast(df, periods=365):
    """Facebook Prophet for time series forecasting"""
    from prophet import Prophet
    
    # df must have 'ds' (date) and 'y' (value) columns
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast

# ============================================================================
# MULTIVARIATE TIME SERIES
# ============================================================================

class MultivariateLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_size, dropout=0.2):
        super(MultivariateLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            num_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, num_features)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

class VectorAutoregression(nn.Module):
    """Neural network version of VAR"""
    def __init__(self, num_features, lag_order, hidden_size=64):
        super(VectorAutoregression, self).__init__()
        
        self.lag_order = lag_order
        input_size = num_features * lag_order
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_features)
        
    def forward(self, x):
        # x: (batch, lag_order, num_features)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================================================================
# ANOMALY DETECTION IN TIME SERIES
# ============================================================================

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for anomaly detection"""
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Repeat hidden state for decoder
        repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decode
        decoded, _ = self.decoder(repeated, (hidden, cell))
        output = self.output_layer(decoded)
        
        return output

def detect_anomalies_with_autoencoder(model, data, threshold_percentile=95):
    """Detect anomalies using reconstruction error"""
    model.eval()
    
    with torch.no_grad():
        reconstructed = model(data)
        mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2))
    
    threshold = np.percentile(mse.cpu().numpy(), threshold_percentile)
    anomalies = mse > threshold
    
    return anomalies, mse

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_timeseries_model(model, train_loader, val_loader, criterion, optimizer,
                           num_epochs=50, device='cuda', early_stopping_patience=10):
    """Training loop for time series models"""
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_ts_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate various forecasting metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # SMAPE (Symmetric MAPE)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'SMAPE': smape
    }

def forecast_accuracy(forecast, actual):
    """Calculate multiple accuracy metrics"""
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100
    me = np.mean(forecast - actual)  # Mean Error
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual) * 100  # Mean Percentage Error
    rmse = np.sqrt(np.mean((forecast - actual) ** 2))  # RMSE
    
    return {
        'MAPE': mape,
        'ME': me,
        'MAE': mae,
        'MPE': mpe,
        'RMSE': rmse
    }

# ============================================================================
# FORECASTING UTILITIES
# ============================================================================

def multi_step_forecast(model, initial_sequence, steps, device='cuda'):
    """Multi-step ahead forecasting"""
    model.eval()
    
    current_sequence = initial_sequence.to(device)
    forecasts = []
    
    with torch.no_grad():
        for _ in range(steps):
            # Predict next step
            prediction = model(current_sequence)
            forecasts.append(prediction.cpu().numpy())
            
            # Update sequence (rolling window)
            new_sequence = torch.cat([current_sequence[:, 1:, :], prediction.unsqueeze(1)], dim=1)
            current_sequence = new_sequence
    
    return np.array(forecasts).squeeze()

def ensemble_forecast(models, data, device='cuda'):
    """Ensemble multiple models for better forecasting"""
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(data.to(device))
            predictions.append(pred.cpu().numpy())
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred

# ============================================================================
# CROSS-VALIDATION FOR TIME SERIES
# ============================================================================

def time_series_cv_split(data, n_splits=5, test_size=0.2):
    """Time series cross-validation splits"""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_index, test_index in tscv.split(data):
        splits.append((train_index, test_index))
    
    return splits

def walk_forward_validation(data, initial_train_size, step_size=1):
    """Walk-forward validation for time series"""
    results = []
    
    for i in range(initial_train_size, len(data) - step_size):
        train = data[:i]
        test = data[i:i+step_size]
        results.append((train, test))
    
    return results
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, input_len, horizon, target_cols=None):
        self.feature_cols = feature_cols
        self.target_cols = target_cols or feature_cols
        self.input_len = input_len
        self.horizon = horizon

        data = torch.tensor(
            df[feature_cols].values,
            dtype=torch.float32
        )

        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-6
        self.data = (data - self.mean) / self.std

        self.target_idx = [
            feature_cols.index(c) for c in self.target_cols
        ]

    def __len__(self):
        return len(self.data) - self.input_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_len]
        y = self.data[
            idx + self.input_len :
            idx + self.input_len + self.horizon,
            self.target_idx
        ]
        return x, y
import torch.nn as nn
import torch

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        """
        h: (B, T, H)
        """
        weights = self.score(h).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return context
class AttnLSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, output_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.attn = TemporalAttention(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * output_dim)
        )

        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        context = self.attn(h)

        out = self.head(context)
        out = out.view(-1, self.horizon, self.output_dim)
        return out, state
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, horizon, output_dim):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.3,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Linear(d_model, horizon * output_dim)
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)

        h = self.encoder(x)
        h = h[:, -1]

        out = self.head(h)
        out = out.view(-1, self.horizon, self.output_dim)
        return out
