# options_trading_bot/config.py

import os
from dotenv import load_dotenv

class Config:
    """Configuration for the trading bot"""
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # API Credentials
        self.TRADIER_API_KEY = os.getenv('TRADIER_API_KEY')
        self.TRADIER_ACCOUNT_ID = os.getenv('TRADIER_ACCOUNT_ID')
        
        # Discord Webhook
        self.DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
        
        # Google Drive settings
        self.GOOGLE_DRIVE_FOLDER = os.getenv('GOOGLE_DRIVE_FOLDER', 'TradingBotData')
        self.GOOGLE_CREDENTIALS_FILE = os.getenv('GOOGLE_CREDENTIALS_FILE', 'google_credentials.json')
        
        # Trading parameters
        self.INITIAL_PORTFOLIO_VALUE = float(os.getenv('INITIAL_PORTFOLIO_VALUE', '10000.0'))
        self.MAX_POSITION_SIZE_PERCENT = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '5.0'))
        self.MIN_PRICE_DIFFERENCE_PERCENT = float(os.getenv('MIN_PRICE_DIFFERENCE_PERCENT', '5.0'))
        
        # Stop loss and take profit settings
        self.BASE_STOP_LOSS_PERCENT = float(os.getenv('BASE_STOP_LOSS_PERCENT', '25.0'))
        self.BASE_TAKE_PROFIT_PERCENT = float(os.getenv('BASE_TAKE_PROFIT_PERCENT', '50.0'))
        self.IV_MULTIPLIER_STOP_LOSS = float(os.getenv('IV_MULTIPLIER_STOP_LOSS', '0.5'))
        self.IV_MULTIPLIER_TAKE_PROFIT = float(os.getenv('IV_MULTIPLIER_TAKE_PROFIT', '1.0'))
        
        # Model weights for option pricing
        self.MODEL_WEIGHTS = {
            'black_scholes': float(os.getenv('WEIGHT_BLACK_SCHOLES', '0.2')),
            'merton_jump': float(os.getenv('WEIGHT_MERTON_JUMP', '0.2')),
            'barone_adesi': float(os.getenv('WEIGHT_BARONE_ADESI', '0.2')),
            'monte_carlo': float(os.getenv('WEIGHT_MONTE_CARLO', '0.2')),
            'binomial': float(os.getenv('WEIGHT_BINOMIAL', '0.2'))
        }
        
        # Validate configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configurations are set"""
        required_vars = ['TRADIER_API_KEY', 'TRADIER_ACCOUNT_ID']
        
        for var in required_vars:
            if not getattr(self, var):
                raise ValueError(f"Required configuration missing: {var}")
        
        # Ensure model weights sum to 1.0
        total_weight = sum(self.MODEL_WEIGHTS.values())
        if abs(total_weight - 1.0) > 0.001:
            # Normalize weights
            for model in self.MODEL_WEIGHTS:
                self.MODEL_WEIGHTS[model] /= total_weight


# Example .env file content (create this file in your project root)
"""
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
"""
