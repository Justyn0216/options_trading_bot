# options_trading_bot/main.py

import time
import logging
from datetime import datetime

from config import Config
from models.option_pricing import OptionPricingEngine
from apis.tradier_api import TradierAPI
from scanner.option_scanner import OptionScanner
from trading.paper_trader import PaperTrader
from notifications.discord_webhook import DiscordNotifier
from data.data_manager import DataManager
from ml.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("options_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Options Trading Bot")
    
    # Initialize components
    config = Config()
    tradier_api = TradierAPI(config.TRADIER_API_KEY, config.TRADIER_ACCOUNT_ID, sandbox_mode=True)
    data_manager = DataManager(config.GOOGLE_DRIVE_FOLDER)
    pricing_engine = OptionPricingEngine()
    discord = DiscordNotifier(config.DISCORD_WEBHOOK_URL)
    
    # Initialize scanner
    scanner = OptionScanner(
        tradier_api=tradier_api,
        pricing_engine=pricing_engine,
        symbol="QQQ",
        min_dte=0,
        max_dte=180
    )
    
    # Initialize paper trader
    paper_trader = PaperTrader(
        tradier_api=tradier_api,
        data_manager=data_manager,
        discord=discord,
        initial_balance=config.INITIAL_PORTFOLIO_VALUE
    )
    
    # Initialize ML model trainer
    model_trainer = ModelTrainer(data_manager)
    
    # Initial model training if historical data exists
    if data_manager.historical_data_exists():
        model_trainer.train_models()
    
    logger.info(f"Starting with portfolio value: ${config.INITIAL_PORTFOLIO_VALUE:.2f}")
    discord.send_message(f"🤖 Options Trading Bot started! Initial portfolio: ${config.INITIAL_PORTFOLIO_VALUE:.2f}")
    
    try:
        run_trading_loop(scanner, paper_trader, model_trainer, data_manager, discord, config)
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
        discord.send_message("🛑 Bot shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        discord.send_message(f"⚠️ Bot encountered an error: {str(e)}")
    finally:
        final_value = paper_trader.get_portfolio_value()
        logger.info(f"Bot stopped. Final portfolio value: ${final_value:.2f}")
        discord.send_message(f"🛑 Bot stopped. Final portfolio value: ${final_value:.2f}")

def run_trading_loop(scanner, paper_trader, model_trainer, data_manager, discord, config):
    """Main trading loop that runs continuously"""
    
    scan_interval = 60  # seconds
    ml_retrain_interval = 10  # number of completed trades before retraining
    completed_trades_since_retrain = 0
    
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Scanning for opportunities at {current_time}")
        
        # Update open positions status
        if paper_trader.has_open_positions():
            paper_trader.update_open_positions()
        
        # Scan for new opportunities
        opportunities = scanner.scan_options()
        
        # Use ML model to filter opportunities if available
        if model_trainer.models_available():
            opportunities = model_trainer.filter_opportunities(opportunities)
        
        # Enter new trades if there are opportunities and no open positions
        if opportunities and not paper_trader.has_open_positions():
            best_opportunity = opportunities[0]  # Get the highest-ranked opportunity
            
            logger.info(f"Found opportunity: {best_opportunity['symbol']} at ${best_opportunity['mark']:.2f}")
            discord.send_message(f"🔍 Found opportunity: {best_opportunity['symbol']} at ${best_opportunity['mark']:.2f}")
            
            # Enter paper trade
            paper_trader.enter_position(best_opportunity)
        
        # Check if any trades were completed
        completed_trades = paper_trader.get_completed_trades_since_last_check()
        if completed_trades:
            completed_trades_since_retrain += len(completed_trades)
            
            # Save trade data
            for trade in completed_trades:
                data_manager.save_trade_data(trade)
            
            # Update portfolio value
            paper_trader.update_portfolio_value()
            current_value = paper_trader.get_portfolio_value()
            data_manager.save_portfolio_value(current_value)
            
            # Retrain ML models if threshold reached
            if completed_trades_since_retrain >= ml_retrain_interval:
                logger.info("Retraining ML models with new trade data")
                discord.send_message("🧠 Retraining ML models with new trade data")
                model_trainer.train_models()
                completed_trades_since_retrain = 0
        
        # Sleep until next scan
        time.sleep(scan_interval)

if __name__ == "__main__":
    main()
