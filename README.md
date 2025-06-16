📂 Project Overview
📡 Data Source: Satellite-based particulate matter data (particulate_matter_2.5um.grib)

🧹 Preprocessing: Extracts, filters, and applies a log transformation on PM2.5 concentration

🧠 Modeling: A CNN is trained to forecast next-timestep PM2.5 concentration based on current observations

📊 Evaluation: Predicted vs actual comparison on random validation samples

🗺️ Visualization: Spatio temporal visualisation of PM2.5 distributions using Cartopy from December 10 to December 15, 2024


📈 Dataset Format
Input Shape: (time, latitude, longitude)

Log transformation applied: log(x + 1e-6) to ensure numerical stability

Output target: PM2.5 concentration at time t+1 for each spatial grid

🧠 Model Architecture
class PM25PredictorCNN(nn.Module):
    def __init__(self):
        ...
Fully convolutional model with 4 layers

Accepts (1, 34, 61) spatial input grids and outputs same shape

Optimized using MSELoss and Adam optimizer

🧪 Training Results
Epoch	Train Loss	Validation Loss
1	13.9810	0.0716
10	0.0008	0.0007

Model learns spatiotemporal patterns well after log-scaling the data

Final validation error is minimal, suggesting strong predictive capability

🌍 Visualizations
Uses Cartopy for geographically-aware PM2.5 plotting

Supports:

Colorbar adjustment

Latitude/longitude labeling

Coastlines and borders

Random sample prediction comparison

📦 Dependencies
xarray
cfgrib
cartopy
matplotlib
pandas
numpy
torch

🔍 Goals & Applications
Build early-warning systems for air quality
Integrate into environmental monitoring dashboards
Extend to multi-step forecasts or other pollutants (e.g., PM10, NO2)
