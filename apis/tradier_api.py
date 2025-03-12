# options_trading_bot/apis/tradier_api.py

import requests
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class TradierAPI:
    """
    Class to interact with the Tradier API for market data and trading operations
    """
    
    def __init__(self, api_key, account_id=None, sandbox_mode=True):
        """
        Initialize the Tradier API client
        
        Args:
            api_key (str): Tradier API key
            account_id (str): Tradier account ID (required for trading operations)
            sandbox_mode (bool): Whether to use the sandbox environment
        """
        self.api_key = api_key
        self.account_id = account_id
        
        # Set base URL based on environment
        if sandbox_mode:
            self.base_url = "https://sandbox.tradier.com/v1"
        else:
            self.base_url = "https://api.tradier.com/v1"
        
        # Set headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 0.2  # seconds between requests to avoid rate limiting
    
    def _make_request(self, method, endpoint, params=None, data=None):
        """
        Make a request to the Tradier API with rate limiting
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            params (dict): Query parameters
            data (dict): Form data for POST requests
            
        Returns:
            dict: Response JSON
        """
        # Rate limiting - ensure minimum interval between requests
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        url = f"{self.base_url}/{endpoint}"
        
        # Make the request
        try:
            self.last_request_time = time.time()
            
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, params=params, data=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            
            # Retry on rate limiting (HTTP 429)
            if hasattr(e, 'response') and e.response.status_code == 429:
                logger.warning("Rate limited. Waiting and retrying...")
                time.sleep(2)  # Wait longer before retry
                return self._make_request(method, endpoint, params, data)
                
            return {"error": str(e)}
    
    def get_quotes(self, symbols):
        """
        Get current quotes for symbols
        
        Args:
            symbols (str or list): Symbol(s) to get quotes for
            
        Returns:
            dict: Quote data
        """
        if isinstance(symbols, list):
            symbols = ",".join(symbols)
            
        return self._make_request("GET", "markets/quotes", params={"symbols": symbols})
    
    def get_option_chains(self, symbol, expiration=None):
        """
        Get option chain for a symbol
        
        Args:
            symbol (str): Underlying symbol
            expiration (str): Expiration date in YYYY-MM-DD format
            
        Returns:
            dict: Option chain data
        """
        params = {"symbol": symbol}
        if expiration:
            params["expiration"] = expiration
            
        return self._make_request("GET", "markets/options/chains", params=params)
    
    def get_option_expirations(self, symbol, include_all_roots=False):
        """
        Get available option expirations for a symbol
        
        Args:
            symbol (str): Underlying symbol
            include_all_roots (bool): Include all option roots
            
        Returns:
            dict: Expiration dates
        """
        params = {
            "symbol": symbol,
            "includeAllRoots": include_all_roots
        }
        
        return self._make_request("GET", "markets/options/expirations", params=params)
    
    def get_option_strikes(self, symbol, expiration):
        """
        Get available strikes for an option
        
        Args:
            symbol (str): Underlying symbol
            expiration (str): Expiration date in YYYY-MM-DD format
            
        Returns:
            dict: Available strikes
        """
        params = {
            "symbol": symbol,
            "expiration": expiration
        }
        
        return self._make_request("GET", "markets/options/strikes", params=params)
    
    def lookup_option_symbols(self, underlying, expiration=None, strike=None, option_type=None):
        """
        Lookup option symbols
        
        Args:
            underlying (str): Underlying symbol
            expiration (str): Expiration date in YYYY-MM-DD format
            strike (float): Strike price
            option_type (str): Option type (call/put)
            
        Returns:
            dict: Option symbols
        """
        params = {"underlying": underlying}
        
        if expiration:
            params["expiration"] = expiration
        if strike:
            params["strike"] = strike
        if option_type:
            params["type"] = option_type
            
        return self._make_request("GET", "markets/options/lookup", params=params)
    
    def build_option_symbol(self, symbol, expiration_date, strike, option_type):
        """
        Build an OCC option symbol
        
        Args:
            symbol (str): Underlying symbol (e.g., 'QQQ')
            expiration_date (str): Expiration date in YYYY-MM-DD format
            strike (float): Strike price
            option_type (str): Option type ('call' or 'put')
            
        Returns:
            str: OCC option symbol
        """
        # Parse expiration date
        expiry = datetime.strptime(expiration_date, "%Y-%m-%d")
        
        # Pad symbol to 6 characters
        padded_symbol = symbol.ljust(6)
        
        # Format expiration as YYMMDD
        expiry_code = expiry.strftime("%y%m%d")
        
        # Format option type (C for call, P for put)
        type_code = "C" if option_type.lower() == "call" else "P"
        
        # Format strike price (multiply by 1000 and remove decimal)
        strike_int = int(strike * 1000)
        strike_code = str(strike_int).zfill(8)
        
        # Build OCC symbol: Symbol + Expiry + Type + Strike
        occ_symbol = f"{padded_symbol}{expiry_code}{type_code}{strike_code}"
        
        return occ_symbol
    
    def parse_option_symbol(self, option_symbol):
        """
        Parse an OCC option symbol into its components
        
        Args:
            option_symbol (str): OCC option symbol
            
        Returns:
            dict: Components of the symbol
        """
        # Extract components
        symbol = option_symbol[:6].strip()
        year = 2000 + int(option_symbol[6:8])
        month = int(option_symbol[8:10])
        day = int(option_symbol[10:12])
        option_type = "call" if option_symbol[12] == "C" else "put"
        strike = float(option_symbol[13:]) / 1000.0
        
        expiration_date = f"{year}-{month:02d}-{day:02d}"
        
        return {
            "symbol": symbol,
            "expiration_date": expiration_date,
            "option_type": option_type,
            "strike": strike
        }
    
    def calculate_time_to_expiry(self, expiration_date):
        """
        Calculate time to expiry in years
        
        Args:
            expiration_date (str): Expiration date in YYYY-MM-DD format
            
        Returns:
            float: Time to expiry in years
        """
        expiry = datetime.strptime(expiration_date, "%Y-%m-%d")
        now = datetime.now()
        
        # Calculate days to expiry
        days_to_expiry = (expiry - now).days + (expiry - now).seconds / 86400.0
        
        # Convert to years (assuming 365 days in a year)
        years_to_expiry = max(days_to_expiry / 365.0, 0.0)
        
        return years_to_expiry
    
    def calculate_days_to_expiry(self, expiration_date):
        """
        Calculate days to expiry
        
        Args:
            expiration_date (str): Expiration date in YYYY-MM-DD format
            
        Returns:
            int: Days to expiry
        """
        expiry = datetime.strptime(expiration_date, "%Y-%m-%d")
        now = datetime.now()
        
        # Calculate days to expiry
        days_to_expiry = (expiry - now).days
        
        return max(days_to_expiry, 0)
    
    def get_option_market_data(self, option_symbol):
        """
        Get full market data for an option symbol
        
        Args:
            option_symbol (str): Option symbol (OCC format)
            
        Returns:
            dict: Option market data including greeks and IV
        """
        # Get quote for the option
        option_quote = self.get_quotes(option_symbol)
        
        # Check for error
        if "error" in option_quote:
            return {"error": option_quote["error"]}
        
        # Extract the quote
        try:
            quote = option_quote.get("quotes", {}).get("quote", {})
            
            # If no quote was found
            if not quote:
                return {"error": "No quote found for option"}
                
            # Parse the option symbol
            option_info = self.parse_option_symbol(option_symbol)
            
            # Get underlying quote
            underlying_quote = self.get_quotes(option_info["symbol"])
            underlying_price = underlying_quote.get("quotes", {}).get("quote", {}).get("last", 0.0)
            
            # Calculate days to expiry
            dte = self.calculate_days_to_expiry(option_info["expiration_date"])
            
            # Calculate time to expiry in years
            time_to_expiry = self.calculate_time_to_expiry(option_info["expiration_date"])
            
            # Extract option data
            result = {
                "symbol": option_symbol,
                "underlying": option_info["symbol"],
                "underlying_price": underlying_price,
                "strike": option_info["strike"],
                "option_type": option_info["option_type"],
                "expiration_date": option_info["expiration_date"],
                "days_to_expiry": dte,
                "time_to_expiry": time_to_expiry,
                "bid": quote.get("bid", 0.0),
                "ask": quote.get("ask", 0.0),
                "last": quote.get("last", 0.0),
                "mark": (quote.get("bid", 0.0) + quote.get("ask", 0.0)) / 2,
                "volume": quote.get("volume", 0),
                "open_interest": quote.get("open_interest", 0),
                "implied_volatility": quote.get("greeks", {}).get("mid_iv", 0.0),
            }
            
            # Add greeks if available
            greeks = quote.get("greeks", {})
            if greeks:
                result.update({
                    "delta": greeks.get("delta", 0.0),
                    "gamma": greeks.get("gamma", 0.0),
                    "theta": greeks.get("theta", 0.0),
                    "vega": greeks.get("vega", 0.0),
                    "rho": greeks.get("rho", 0.0)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing option market data: {e}")
            return {"error": str(e)}
