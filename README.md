# Options Trading Bot

A real-time machine learning & reinforcement learning options scanner and paper trading bot that uses multiple option pricing models to find undervalued options and execute trades.

## Features

- **Multi-Model Option Pricing**: Utilizes Black-Scholes, Merton Jump Diffusion, Barone-Adesi Whaley, Monte Carlo, and Binomial models with weighted averaging
- **Real-Time Scanning**: Scans the option chain for QQQ every 60 seconds to find undervalued options
- **ML-Powered Trading**: Uses machine learning to improve trade selection based on historical performance
- **Dynamic Risk Management**: Sets dynamic stop losses and take profits based on implied volatility
- **Paper Trading**: Simulates trades without risking real money
- **Detailed Logging**: Records trades, performance, and portfolio value
- **Discord Notifications**: Get real-time alerts via Discord webhook
- **Google Drive Integration**: Stores all trading data in your Google Drive for persistence and analysis

## Requirements

- Python 3.8+
- Tradier brokerage account (sandbox or live)
- Discord webhook URL (for notifications)
- Google Drive API credentials (for data storage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/options-trading-bot.git
cd options-trading-bot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.template .env
```

4. Edit the `.env` file with your credentials:
```
# API Credentials
TRADIER_API_KEY=your_tradier_api_key
TRADIER_ACCOUNT_ID=your_tradier_account_id

# Discord Webhook
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Google Drive
GOOGLE_DRIVE_FOLDER=TradingBotData
GOOGLE_CREDENTIALS_FILE=google_credentials.json

# Trading parameters
INITIAL_PORTFOLIO_VALUE=10000.0
MAX_POSITION_SIZE_PERCENT=5.0
MIN_PRICE_DIFFERENCE_PERCENT=5.0

# Stop loss and take profit
BASE_STOP_LOSS_PERCENT=25.0
BASE_TAKE_PROFIT_PERCENT=50.0
IV_MULTIPLIER_STOP_LOSS=0.5
IV_MULTIPLIER_TAKE_PROFIT=1.0

# Model weights
WEIGHT_BLACK_SCHOLES=0.2
WEIGHT_MERTON_JUMP=0.2
WEIGHT_BARONE_ADESI=0.2
WEIGHT_MONTE_CARLO=0.2
WEIGHT_BINOMIAL=0.2
```

5. Set up Google Drive API access:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable the Google Drive API
   - Create a service account
   - Download the service account credentials as JSON
   - Save the JSON file as `google_credentials.json` in the project root

## Project Structure

```
options_trading_bot/
├── main.py                  # Main application entry point
├── config.py                # Configuration loader
├── models/
│   └── option_pricing.py    # Option pricing models
├── apis/
│   └── tradier_api.py       # Tradier API client
├── scanner/
│   └── option_scanner.py    # Options scanning logic
├── trading/
│   └── paper_trader.py      # Paper trading system
├── notifications/
│   └── discord_webhook.py   # Discord notification system
├── data/
│   └── data_manager.py      # Data storage and retrieval
├── ml/
│   └── model_trainer.py     # Machine learning model
└── data/                    # Local data storage
    ├── tradingbot_trades.csv
    └── tradingbot_portfolio_v1.csv
```

## Usage

1. Run the bot:
```bash
python -m options_trading_bot.main
```

2. Monitor the notifications in your Discord channel

3. Check the trading data in your Google Drive folder "TradingBotData"

## Customization

### Option Pricing Model Weights

You can adjust the weights of the different option pricing models in the `.env` file:

```
WEIGHT_BLACK_SCHOLES=0.2
WEIGHT_MERTON_JUMP=0.2
WEIGHT_BARONE_ADESI=0.2
WEIGHT_MONTE_CARLO=0.2
WEIGHT_BINOMIAL=0.2
```

The weights should sum to 1.0. If not, they will be automatically normalized.

### Risk Management Parameters

Customize stop loss and take profit levels:

```
BASE_STOP_LOSS_PERCENT=25.0      # Base stop loss percentage
BASE_TAKE_PROFIT_PERCENT=50.0    # Base take profit percentage
IV_MULTIPLIER_STOP_LOSS=0.5      # How much IV affects stop loss width
IV_MULTIPLIER_TAKE_PROFIT=1.0    # How much IV affects take profit target
```

### Position Sizing

Control how much of your portfolio is risked on each trade:

```
INITIAL_PORTFOLIO_VALUE=10000.0   # Starting portfolio value
MAX_POSITION_SIZE_PERCENT=5.0     # Maximum size of any position
```

## Limitations and Disclaimer

This is a paper trading bot for educational purposes only. It does not execute real trades. Performance in paper trading is not indicative of future results with real money.

The machine learning models require sufficient historical data to make useful predictions. The bot will need to run for some time to gather enough data for the ML component to become effective.

## License

MIT
