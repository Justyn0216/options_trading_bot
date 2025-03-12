# options_trading_bot/notifications/discord_webhook.py

import requests
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DiscordNotifier:
    """
    Sends notifications to Discord via webhook
    """
    
    def __init__(self, webhook_url=None, username="Options Trading Bot", avatar_url=None):
        """
        Initialize Discord notifier
        
        Args:
            webhook_url (str): Discord webhook URL
            username (str): Bot username
            avatar_url (str): URL to avatar image
        """
        self.webhook_url = webhook_url
        self.username = username
        self.avatar_url = avatar_url
        
        if not webhook_url:
            logger.warning("Discord webhook URL not provided. Notifications will be logged only.")
    
    def send_message(self, content, embed=None):
        """
        Send message to Discord webhook
        
        Args:
            content (str): Message content
            embed (dict): Optional embed data
            
        Returns:
            bool: True if message was sent successfully
        """
        # Always log the message
        logger.info(f"Discord notification: {content}")
        
        if not self.webhook_url:
            return False
            
        try:
            payload = {
                "content": content,
                "username": self.username
            }
            
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
                
            if embed:
                payload["embeds"] = [embed]
                
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers=headers
            )
            
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    def send_trade_alert(self, trade_data, action="ALERT"):
        """
        Send formatted trade alert
        
        Args:
            trade_data (dict): Trade data
            action (str): Alert action type
            
        Returns:
            bool: True if alert was sent successfully
        """
        if not self.webhook_url:
            return False
            
        try:
            # Format timestamp
            timestamp = datetime.now().isoformat()
            
            # Determine color based on action
            color_map = {
                "ENTRY": 0x3498db,  # Blue
                "EXIT": 0x2ecc71,   # Green
                "ALERT": 0xf1c40f,  # Yellow
                "ERROR": 0xe74c3c   # Red
            }
            color = color_map.get(action.upper(), 0xf1c40f)
            
            # Create embed
            embed = {
                "title": f"{action}: {trade_data['symbol']}",
                "color": color,
                "timestamp": timestamp,
                "fields": []
            }
            
            # Add fields based on available data
            if "option_type" in trade_data and "strike" in trade_data:
                embed["fields"].append({
                    "name": "Option",
                    "value": f"{trade_data['option_type'].upper()} ${trade_data['strike']}",
                    "inline": True
                })
                
            if "days_to_expiry" in trade_data:
                embed["fields"].append({
                    "name": "DTE",
                    "value": f"{trade_data['days_to_expiry']}",
                    "inline": True
                })
                
            if "entry_price" in trade_data:
                embed["fields"].append({
                    "name": "Entry",
                    "value": f"${trade_data['entry_price']:.2f}",
                    "inline": True
                })
                
            if "current_price" in trade_data:
                embed["fields"].append({
                    "name": "Current",
                    "value": f"${trade_data['current_price']:.2f}",
                    "inline": True
                })
                
            if "exit_price" in trade_data:
                embed["fields"].append({
                    "name": "Exit",
                    "value": f"${trade_data['exit_price']:.2f}",
                    "inline": True
                })
                
            if "pnl_percent" in trade_data:
                embed["fields"].append({
                    "name": "P&L %",
                    "value": f"{trade_data['pnl_percent']:.2f}%",
                    "inline": True
                })
                
            # Send webhook
            return self.send_message("", embed=embed)
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
            return False
