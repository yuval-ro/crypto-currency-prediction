# Cryptocurrency Predictor

## Powered by

[![frameworks](https://skillicons.dev/icons?i=python,pytorch,sklearn,docker&theme=light)](https://skillicons.dev)
[![PDM](https://img.shields.io/badge/PDM-managed-blueviolet)](https://pdm.fming.dev)

## About

Deep learning cryptocurrency price prediction using a hybrid LSTM-Linear model. Combines PyTorch's neural networks with scikit-learn's intuitive API design for time-series forecasting with real-world data.

## Overview

<img src=".github/flowchart.svg"/>

## Features

### Real-Time Data Fetching

- **CryptoCompare API Integration** for historical OHLCV data
- **10+ Cryptocurrencies**: BTC, ETH, XRP, USDT, USDC, BUSD, BNB, TUSD, DOGE, SHIB
- **Multi-Currency Support**: USD, EUR, and more

### Scikit-learn Inspired API

Clean, intuitive interface that abstracts PyTorch complexity:

```python
generator = DatasetGenerator('btc', 'usd')
train_ds, valid_ds, test_ds = generator.split()

model = Model(generator.dataset)
trainer = ModelTrainer(model)
trainer.train(train_ds, valid_ds, hp)
trainer.test(test_ds, 'predictions.png')
```

### Hybrid Architecture

- **LSTM layers** capture temporal patterns
- **Linear regression head** for price predictions
- **Configurable**: 2-layer LSTM with 256 hidden units

### Production Features

- **Early Stopping** with configurable tolerance
- **Automatic MinMax Scaling** for normalization
- **Loss Visualization** with training/validation curves
- **Model Checkpointing** for save/load

## Installation

```bash
# Build and run with Docker using PDM
pdm deploy

# Or manually:
docker build -t crypto-predictor .
docker run -v $(pwd)/outputs:/app/outputs crypto-predictor

# Outputs saved to ./outputs/
```

## Legal

Licensed under [GPLv3](./LICENSE). Copyright (C) 2025 yuval-ro.

**Note**: For educational purposes only. Not financial advice.
