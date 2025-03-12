# options_trading_bot/trading/paper_trader.py

import logging
import uuid
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class PaperTrader:
    """
    Paper trading system for options
    """
    
    def __init__(self, tradier_api, data_manager, discord, initial_balance=10000.0, config=None):
        """
        Initialize paper trader
        
        Args:
            tradier_api: TradierAPI instance
            data_manager: DataManager instance
            discord: DiscordNotifier instance
            initial_balance (float): Initial portfolio value
            config: Configuration object
        """
        self.tradier_api = tradier_api
        self.data_manager = data_manager
        self.discord = discord
        self.config = config
        
        # Trading parameters
        if config:
            self.max_position_size_percent = config.MAX_POSITION_SIZE_PERCENT / 100.0
            self.base_stop_loss_percent = config.BASE_STOP_LOSS_PERCENT / 100.0
            self.base_take_profit_percent = config.BASE_TAKE_PROFIT_PERCENT / 100.0
            self.iv_multiplier_stop_loss = config.IV_MULTIPLIER_STOP_LOSS
            self.iv_multiplier_take_profit = config.IV_MULTIPLIER_TAKE_PROFIT
        else:
            # Default values if no config provided
            self.max_position_size_percent = 0.05  # 5% of portfolio
            self.base_stop_loss_percent = 0.25     # 25% loss
            self.base_take_profit_percent = 0.50   # 50% gain
            self.iv_multiplier_stop_loss = 0.5
            self.iv_multiplier_take_profit = 1.0
        
        # Portfolio tracking
        self.portfolio_value = initial_balance
        self.initial_portfolio_value = initial_balance
        
        # Open and closed positions
        self.open_positions = {}
        self.closed_positions = []
        self.completed_trades_since_last_check = []
        
        # Load existing portfolio value if available
        self._load_portfolio_value()
    
    def _load_portfolio_value(self):
        """Load portfolio value from saved data if available"""
        try:
            portfolio_data = self.data_manager.load_portfolio_value()
            if portfolio_data and portfolio_data.get('value', 0) > 0:
                self.portfolio_value = portfolio_data.get('value')
                logger.info(f"Loaded portfolio value: ${self.portfolio_value:.2f}")
        except Exception as e:
            logger.error(f"Error loading portfolio value: {e}")
    
    def _calculate_dynamic_stop_loss(self, entry_price, option_type, iv):
        """
        Calculate dynamic stop loss based on IV
        
        Args:
            entry_price (float): Entry price
            option_type (str): 'call' or 'put'
            iv (float): Implied volatility
            
        Returns:
            float: Stop loss price
        """
        # Adjust stop loss based on IV - higher IV means wider stop
        stop_loss_percent = self.base_stop_loss_percent * (1 + iv * self.iv_multiplier_stop_loss)
        
        if option_type == 'call':
            return entry_price * (1 - stop_loss_percent)
        else:  # put
            return entry_price * (1 - stop_loss_percent)
    
    def _calculate_dynamic_take_profit(self, entry_price, option_type, iv):
        """
        Calculate dynamic take profit based on IV
        
        Args:
            entry_price (float): Entry price
            option_type (str): 'call' or 'put'
            iv (float): Implied volatility
            
        Returns:
            float: Take profit price
        """
        # Adjust take profit based on IV - higher IV means higher target
        take_profit_percent = self.base_take_profit_percent * (1 + iv * self.iv_multiplier_take_profit)
        
        if option_type == 'call':
            return entry_price * (1 + take_profit_percent)
        else:  # put
            return entry_price * (1 + take_profit_percent)
    
    def enter_position(self, option_data):
        """
        Enter a paper trade position
        
        Args:
            option_data (dict): Option data
            
        Returns:
            dict: Position data
        """
        # Calculate position size
        max_position_value = self.portfolio_value * self.max_position_size_percent
        contract_price = option_data["mark"]
        contract_multiplier = 100  # Standard for options
        
        # Calculate how many contracts we can buy
        max_contracts = int(max_position_value / (contract_price * contract_multiplier))
        
        # Ensure at least 1 contract if portfolio allows
        quantity = max(1, max_contracts)
        
        # Ensure we don't exceed portfolio value
        total_cost = quantity * contract_price * contract_multiplier
        if total_cost > self.portfolio_value:
            logger.warning(f"Position cost exceeds portfolio value. Adjusting quantity.")
            quantity = max(1, int(self.portfolio_value / (contract_price * contract_multiplier)))
            total_cost = quantity * contract_price * contract_multiplier
        
        # Calculate dynamic stop loss and take profit
        iv = option_data["implied_volatility"]
        stop_loss = self._calculate_dynamic_stop_loss(contract_price, option_data["option_type"], iv)
        take_profit = self._calculate_dynamic_take_profit(contract_price, option_data["option_type"], iv)
        
        # Create position
        position_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        position = {
            "id": position_id,
            "symbol": option_data["symbol"],
            "underlying": option_data["underlying"],
            "option_type": option_data["option_type"],
            "strike": option_data["strike"],
            "expiration_date": option_data["expiration_date"],
            "days_to_expiry": option_data["days_to_expiry"],
            "entry_time": timestamp,
            "entry_price": contract_price,
            "quantity": quantity,
            "total_cost": total_cost,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "iv_at_entry": iv,
            "theoretical_price": option_data.get("theoretical_price", 0.0),
            "price_difference_percent": option_data.get("price_difference_percent", 0.0),
            "status": "open",
            "valuation": "undervalued",
            "current_price": contract_price,
            "current_time": timestamp,
            "delta": option_data.get("delta", 0.0),
            "gamma": option_data.get("gamma", 0.0),
            "theta": option_data.get("theta", 0.0),
            "vega": option_data.get("vega", 0.0),
            "rho": option_data.get("rho", 0.0),
            "pnl": 0.0,
            "pnl_percent": 0.0
        }
        
        # Store position
        self.open_positions[position_id] = position
        
        # Log entry
        logger.info(f"Entered position: {position['symbol']} x {quantity} @ ${contract_price:.2f}")
        
        # Send notification
        self.discord.send_message(
            f"🚀 Entered position: {position['symbol']} ({position['option_type'].upper()}) x {quantity} @ ${contract_price:.2f}\n"
            f"Strike: ${position['strike']:.2f}, Expiry: {position['expiration_date']} ({position['days_to_expiry']} DTE)\n"
            f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}\n"
            f"IV: {iv:.2f}, Delta: {position['delta']:.4f}, Theta: {position['theta']:.4f}"
        )
        
        return position
    
    def update_open_positions(self):
        """
        Update all open positions with current market data
        """
        if not self.open_positions:
            return
            
        logger.debug(f"Updating {len(self.open_positions)} open positions")
        
        positions_to_close = []
        updated_positions = []
        
        for position_id, position in self.open_positions.items():
            # Get current option data
            option_data = self.tradier_api.get_option_market_data(position["symbol"])
            
            if "error" in option_data:
                logger.error(f"Error updating position {position_id}: {option_data['error']}")
                continue
            
            # Update position with current market data
            prev_price = position["current_price"]
            prev_valuation = position["valuation"]
            
            # Update current price (use mark price)
            current_price = (option_data["bid"] + option_data["ask"]) / 2 if option_data["bid"] > 0 and option_data["ask"] > 0 else option_data["last"]
            position["current_price"] = current_price
            position["current_time"] = datetime.now().isoformat()
            
            # Update Greeks
            position["delta"] = option_data.get("delta", position["delta"])
            position["gamma"] = option_data.get("gamma", position["gamma"])
            position["theta"] = option_data.get("theta", position["theta"])
            position["vega"] = option_data.get("vega", position["vega"])
            position["rho"] = option_data.get("rho", position["rho"])
            
            # Calculate current P&L
            position["pnl"] = (current_price - position["entry_price"]) * position["quantity"] * 100
            if position["entry_price"] > 0:
                position["pnl_percent"] = (current_price - position["entry_price"]) / position["entry_price"] * 100
            
            # Update theoretical price and valuation
            if "theoretical_price" in option_data:
                position["theoretical_price"] = option_data["theoretical_price"]
                
                # Determine if option is undervalued, fairly valued, or overpriced
                price_diff_percent = (position["theoretical_price"] - current_price) / current_price * 100 if current_price > 0 else 0
                
                if price_diff_percent >= 5.0:
                    new_valuation = "undervalued"
                elif price_diff_percent <= -5.0:
                    new_valuation = "overpriced"
                else:
                    new_valuation = "fair_value"
                
                position["valuation"] = new_valuation
                
                # Notify if valuation changed
                if new_valuation != prev_valuation:
                    updated_positions.append({
                        "symbol": position["symbol"],
                        "prev_valuation": prev_valuation,
                        "new_valuation": new_valuation,
                        "price": current_price,
                        "pnl_percent": position["pnl_percent"]
                    })
            
            # Check if we need to close the position
            should_close = False
            close_reason = ""
            
            # Check stop loss
            if current_price <= position["stop_loss"]:
                should_close = True
                close_reason = "stop_loss"
            
            # Check take profit
            elif current_price >= position["take_profit"]:
                should_close = True
                close_reason = "take_profit"
            
            # Check if near expiration (within 1 day) to avoid expiration risk
            elif position["days_to_expiry"] <= 1:
                should_close = True
                close_reason = "near_expiration"
            
            # If position should be closed, add to list
            if should_close:
                positions_to_close.append((position_id, close_reason))
        
        # Send notifications for valuation changes
        for position in updated_positions:
            valuation_emoji = {
                "undervalued": "🟢",
                "fair_value": "⚪",
                "overpriced": "🔴"
            }
            
            prev_emoji = valuation_emoji.get(position["prev_valuation"], "⚪")
            new_emoji = valuation_emoji.get(position["new_valuation"], "⚪")
            
            self.discord.send_message(
                f"📊 Valuation Change: {position['symbol']}\n"
                f"Status: {prev_emoji} {position['prev_valuation']} → {new_emoji} {position['new_valuation']}\n"
                f"Price: ${position['price']:.2f}, P&L: {position['pnl_percent']:.2f}%"
            )
        
        # Close positions that hit stop loss or take profit
        for position_id, reason in positions_to_close:
            self.close_position(position_id, reason)
    
    def close_position(self, position_id, reason):
        """
        Close a paper trade position
        
        Args:
            position_id (str): Position ID
            reason (str): Reason for closing
            
        Returns:
            dict: Closed position data
        """
        if position_id not in self.open_positions:
            logger.warning(f"Attempted to close non-existent position: {position_id}")
            return None
        
        # Get position
        position = self.open_positions[position_id]
        
        # Record exit details
        position["exit_time"] = datetime.now().isoformat()
        position["exit_price"] = position["current_price"]
        position["exit_reason"] = reason
        position["status"] = "closed"
        
        # Calculate final P&L
        position["pnl"] = (position["exit_price"] - position["entry_price"]) * position["quantity"] * 100
        position["pnl_percent"] = (position["exit_price"] - position["entry_price"]) / position["entry_price"] * 100
        
        # Update portfolio value
        self.portfolio_value += position["pnl"]
        
        # Move to closed positions
        self.closed_positions.append(position)
        self.completed_trades_since_last_check.append(position)
        del self.open_positions[position_id]
        
        # Log exit
        logger.info(
            f"Closed position: {position['symbol']} x {position['quantity']} @ ${position['exit_price']:.2f}, "
            f"P&L: ${position['pnl']:.2f} ({position['pnl_percent']:.2f}%), Reason: {reason}"
        )
        
        # Send notification
        emoji = "🔴" if position["pnl"] < 0 else "🟢"
        self.discord.send_message(
            f"{emoji} Closed position: {position['symbol']} ({position['option_type'].upper()}) x {position['quantity']}\n"
            f"Entry: ${position['entry_price']:.2f}, Exit: ${position['exit_price']:.2f}\n"
            f"P&L: ${position['pnl']:.2f} ({position['pnl_percent']:.2f}%)\n"
            f"Reason: {reason}, Portfolio: ${self.portfolio_value:.2f}"
        )
        
        return position
    
    def has_open_positions(self):
        """
        Check if there are any open positions
        
        Returns:
            bool: True if there are open positions
        """
        return len(self.open_positions) > 0
    
    def get_portfolio_value(self):
        """
        Get current portfolio value
        
        Returns:
            float: Portfolio value
        """
        return self.portfolio_value
    
    def update_portfolio_value(self):
        """
        Update and save portfolio value
        """
        # Save current portfolio value
        self.data_manager.save_portfolio_value(self.portfolio_value)
        logger.info(f"Updated portfolio value: ${self.portfolio_value:.2f}")
        
        return self.portfolio_value
    
    def get_completed_trades_since_last_check(self):
        """
        Get trades completed since last check
        
        Returns:
            list: Completed trades
        """
        completed = self.completed_trades_since_last_check.copy()
        self.completed_trades_since_last_check = []
        return completed
    
    def get_open_positions(self):
        """
        Get all open positions
        
        Returns:
            dict: Open positions
        """
        return self.open_positions
    
    def get_position(self, position_id):
        """
        Get a specific position
        
        Args:
            position_id (str): Position ID
            
        Returns:
            dict: Position data or None if not found
        """
        return self.open_positions.get(position_id) or next(
            (pos for pos in self.closed_positions if pos["id"] == position_id), 
            None
        )
