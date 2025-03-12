# options_trading_bot/scanner/option_scanner.py

import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class OptionScanner:
    """
    Scans option chains to find trading opportunities
    """
    
    def __init__(self, tradier_api, pricing_engine, symbol="QQQ", min_dte=0, max_dte=180, min_volume=10, config=None):
        """
        Initialize scanner
        
        Args:
            tradier_api: TradierAPI instance
            pricing_engine: OptionPricingEngine instance
            symbol (str): Symbol to scan
            min_dte (int): Minimum days to expiration
            max_dte (int): Maximum days to expiration
            min_volume (int): Minimum option volume
            config: Configuration object
        """
        self.tradier_api = tradier_api
        self.pricing_engine = pricing_engine
        self.symbol = symbol
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.min_volume = min_volume
        self.config = config
        
        # Model weights for pricing models
        self.model_weights = None
        if config:
            self.model_weights = config.MODEL_WEIGHTS
            self.min_price_difference_percent = config.MIN_PRICE_DIFFERENCE_PERCENT
        else:
            # Default weights if no config provided
            self.model_weights = {
                'black_scholes': 0.2,
                'merton_jump': 0.2,
                'barone_adesi': 0.2,
                'monte_carlo': 0.2,
                'binomial': 0.2
            }
            self.min_price_difference_percent = 5.0
    
    def scan_options(self):
        """
        Scan option chain for undervalued options
        
        Returns:
            list: Sorted list of undervalued options
        """
        logger.info(f"Scanning options for {self.symbol}")
        
        try:
            # Get available expirations
            expirations_response = self.tradier_api.get_option_expirations(self.symbol)
            
            if "error" in expirations_response:
                logger.error(f"Error getting expirations: {expirations_response['error']}")
                return []
            
            # Extract expiration dates
            expirations = expirations_response.get("expirations", {}).get("date", [])
            if not expirations:
                logger.warning(f"No expirations found for {self.symbol}")
                return []
            
            # Convert to list if it's a single item
            if isinstance(expirations, str):
                expirations = [expirations]
                
            # Filter expirations by DTE
            filtered_expirations = []
            for expiration in expirations:
                dte = self.tradier_api.calculate_days_to_expiry(expiration)
                if self.min_dte <= dte <= self.max_dte:
                    filtered_expirations.append(expiration)
            
            logger.info(f"Found {len(filtered_expirations)} expirations within DTE range {self.min_dte}-{self.max_dte}")
            
            # Get underlying price
            underlying_quote = self.tradier_api.get_quotes(self.symbol)
            underlying_price = underlying_quote.get("quotes", {}).get("quote", {}).get("last", 0.0)
            
            if underlying_price <= 0:
                logger.error(f"Invalid underlying price for {self.symbol}: {underlying_price}")
                return []
                
            # Scan each expiration
            opportunities = []
            
            for expiration in filtered_expirations:
                logger.debug(f"Scanning expiration {expiration}")
                
                # Get option chain for this expiration
                chain_response = self.tradier_api.get_option_chains(self.symbol, expiration)
                
                if "error" in chain_response:
                    logger.error(f"Error getting option chain: {chain_response['error']}")
                    continue
                
                # Extract options
                options = chain_response.get("options", {}).get("option", [])
                if not options:
                    logger.debug(f"No options found for {self.symbol} on {expiration}")
                    continue
                
                # Convert to list if it's a single item
                if isinstance(options, dict):
                    options = [options]
                
                # Process each option
                for option in options:
                    # Skip if volume is too low
                    if option.get("volume", 0) < self.min_volume:
                        continue
                    
                    # Prepare option data for pricing model
                    option_data = {
                        "symbol": option.get("symbol"),
                        "underlying": self.symbol,
                        "underlying_price": underlying_price,
                        "strike": option.get("strike"),
                        "option_type": option.get("option_type").lower(),
                        "expiration_date": option.get("expiration_date"),
                        "days_to_expiry": self.tradier_api.calculate_days_to_expiry(option.get("expiration_date")),
                        "time_to_expiry": self.tradier_api.calculate_time_to_expiry(option.get("expiration_date")),
                        "bid": option.get("bid", 0.0),
                        "ask": option.get("ask", 0.0),
                        "last": option.get("last", 0.0),
                        "mark": (option.get("bid", 0.0) + option.get("ask", 0.0)) / 2,
                        "volume": option.get("volume", 0),
                        "open_interest": option.get("open_interest", 0),
                        "implied_volatility": option.get("greeks", {}).get("mid_iv", 0.0),
                        "is_american": True  # Most equity options are American-style
                    }
                    
                    # Skip if missing critical data
                    if (option_data["implied_volatility"] <= 0 or 
                        option_data["time_to_expiry"] <= 0 or
                        option_data["mark"] <= 0):
                        continue
                    
                    # Calculate theoretical price
                    theoretical_price = self.pricing_engine.calculate_weighted_price(
                        option_data, 
                        weights=self.model_weights
                    )
                    
                    if theoretical_price is None:
                        continue
                    
                    # Calculate price difference (as percentage)
                    if option_data["mark"] > 0:
                        price_difference_percent = (theoretical_price - option_data["mark"]) / option_data["mark"] * 100
                    else:
                        continue
                    
                    # Add theoretical price and difference to option data
                    option_data["theoretical_price"] = theoretical_price
                    option_data["price_difference"] = theoretical_price - option_data["mark"]
                    option_data["price_difference_percent"] = price_difference_percent
                    
                    # Calculate Greeks if not provided by API
                    if "delta" not in option_data:
                        greeks = self.pricing_engine.calculate_greeks(option_data)
                        option_data.update(greeks)
                    
                    # Check if undervalued enough
                    if price_difference_percent >= self.min_price_difference_percent:
                        opportunities.append(option_data)
            
            # Sort opportunities by price difference percentage (highest first)
            sorted_opportunities = sorted(
                opportunities,
                key=lambda x: x["price_difference_percent"],
                reverse=True
            )
            
            logger.info(f"Found {len(sorted_opportunities)} undervalued options")
            
            return sorted_opportunities
            
        except Exception as e:
            logger.error(f"Error scanning options: {e}", exc_info=True)
            return []
    
    def get_detailed_option_data(self, option_symbol):
        """
        Get detailed data for a specific option
        
        Args:
            option_symbol (str): Option symbol
            
        Returns:
            dict: Detailed option data
        """
        return self.tradier_api.get_option_market_data(option_symbol)
    
    def update_model_weights(self, new_weights):
        """
        Update pricing model weights
        
        Args:
            new_weights (dict): New model weights
        """
        # Validate weights
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            # Normalize weights
            for model in new_weights:
                new_weights[model] /= total_weight
        
        self.model_weights = new_weights
        logger.info(f"Updated model weights: {self.model_weights}")
