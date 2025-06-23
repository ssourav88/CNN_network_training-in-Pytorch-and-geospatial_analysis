# About the project
This script utilizes a Feedforward Neural Network (FNN) Convolutional Neural Network (CNN) implemented with the PyTorch library in Python to train and validate a model for predicting next-day concentrations of particulate matter (PM2.5 µm). It also includes spatio-temporal visualizations to illustrate the distribution of PM2.5 across both space and time. 
# 📂 Project Overview
📡 Data Source: Satellite-based particulate matter data (particulate_matter_2.5um.grib)

🧹 Preprocessing: Extracts, filters, and applies a log transformation on PM2.5 concentration

🧠 Modeling: A FNN and CNN model is trained to predict the PM2.5 concentration based on current observations and later used to forecast the next 24 hour PM2.5 concentration.

📊 Evaluation: Predicted vs actual comparison on random validation samples

🗺️ Visualization: Spatio temporal visualisation of PM2.5 distributions using Cartopy from December 10 to December 15, 2024


# 📈 Dataset Format
Input Shape: (time, latitude, longitude)

Log transformation applied: log(x + 1e-6) to ensure numerical stability

Output target: PM2.5 concentration at time t+1 for each spatial grid

# 🧠 Model Architecture
```python
# 1) Feedforward NN (flatten input)
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# 2) Pure CNN Model
class PureCNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 16, 3, padding=1)
        self.conv_out = nn.Conv1d(16, 1, kernel_size=1)  # Predict single value per seq position

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)        # -> (batch, features, seq_len)
        x = self.relu(self.conv1(x)) # -> (batch, 32, seq_len)
        x = self.relu(self.conv2(x)) # -> (batch, 16, seq_len)
        x = self.conv_out(x)         # -> (batch, 1, seq_len)
        x = x.squeeze(1)             # -> (batch, seq_len)
        return x
```
Fully convolutional model with 4 layers

Accepts (1, 34, 61) spatial input grids and outputs same shape

Optimized using MSELoss and Adam optimizer

# 🧪 Training Results
Training Feedforward NN...
Epoch 1 | Train Loss: 0.0049 | Val Loss: 0.0011 |
Epoch 2 | Train Loss: 0.0023 | Val Loss: 0.0010 |
Epoch 3 | Train Loss: 0.0020 | Val Loss: 0.0008 |
Epoch 4 | Train Loss: 0.0017 | Val Loss: 0.0008 |
Epoch 5 | Train Loss: 0.0015 | Val Loss: 0.0008 |
Epoch 6 | Train Loss: 0.0014 | Val Loss: 0.0008 |
Epoch 7 | Train Loss: 0.0014 | Val Loss: 0.0009 |
Epoch 8 | Train Loss: 0.0013 | Val Loss: 0.0007 |
Epoch 9 | Train Loss: 0.0013 | Val Loss: 0.0008 |
Epoch 10 | Train Loss: 0.0013 | Val Loss: 0.0009 |
Epoch 11 | Train Loss: 0.0012 | Val Loss: 0.0008 |
Epoch 12 | Train Loss: 0.0012 | Val Loss: 0.0008 |
Epoch 13 | Train Loss: 0.0012 | Val Loss: 0.0013 |
Epoch 14 | Train Loss: 0.0012 | Val Loss: 0.0007 |
Epoch 15 | Train Loss: 0.0011 | Val Loss: 0.0008 |
Epoch 16 | Train Loss: 0.0011 | Val Loss: 0.0007 |
Epoch 17 | Train Loss: 0.0011 | Val Loss: 0.0007 |
Epoch 18 | Train Loss: 0.0011 | Val Loss: 0.0008 |
Epoch 19 | Train Loss: 0.0011 | Val Loss: 0.0007 |
Epoch 20 | Train Loss: 0.0011 | Val Loss: 0.0010 |
Feedforward NN R2: 0.9957


Model learns spatiotemporal patterns well after log-scaling the data

Final validation error is minimal, suggesting strong predictive capability

# 🌍 Visualizations
Uses Cartopy for geographically-aware PM2.5 plotting
Supports:
- Colorbar adjustment
- Latitude/longitude labeling
- Coastlines and borders
- Random sample prediction comparison

# 📦 Dependencies
xarray,
cfgrib,
cartopy,
matplotlib,
pandas,
numpy,
torch

# 🔍 Goals & Applications
- Build early-warning systems for air quality
- Integrate into environmental monitoring dashboards
- Extend to multi-step forecasts or other pollutants (e.g., PM10, NO2)
