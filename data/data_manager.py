# options_trading_bot/data/data_manager.py

import os
import json
import pandas as pd
import logging
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages data storage and retrieval from Google Drive
    """
    
    def __init__(self, folder_name="TradingBotData", credentials_file="google_credentials.json"):
        """
        Initialize data manager
        
        Args:
            folder_name (str): Google Drive folder name
            credentials_file (str): Path to Google service account credentials file
        """
        self.folder_name = folder_name
        self.credentials_file = credentials_file
        
        # Local data directory
        self.local_data_dir = "data"
        
        # File names
        self.trades_file = "tradingbot_trades.csv"
        self.portfolio_file = "tradingbot_portfolio_v1.csv"
        
        # Initialize
        self._init_directories()
        self.drive_service = self._init_drive_service()
        self.folder_id = self._get_or_create_folder()
    
    def _init_directories(self):
        """Create local data directories if they don't exist"""
        if not os.path.exists(self.local_data_dir):
            os.makedirs(self.local_data_dir)
            logger.info(f"Created local data directory: {self.local_data_dir}")
    
    def _init_drive_service(self):
        """
        Initialize Google Drive API service
        
        Returns:
            Google Drive service or None if not available
        """
        try:
            if not os.path.exists(self.credentials_file):
                logger.warning(f"Google credentials file not found: {self.credentials_file}")
                return None
                
            scopes = ['https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_file(
                self.credentials_file, 
                scopes=scopes
            )
            
            service = build('drive', 'v3', credentials=credentials)
            logger.info("Google Drive API service initialized")
            return service
            
        except Exception as e:
            logger.error(f"Error initializing Google Drive service: {e}")
            return None
    
    def _get_or_create_folder(self):
        """
        Get or create Google Drive folder
        
        Returns:
            str: Folder ID or None if unavailable
        """
        if not self.drive_service:
            return None
            
        try:
            # Check if folder exists
            query = f"name='{self.folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            response = self.drive_service.files().list(q=query, spaces='drive').execute()
            
            if response.get('files'):
                folder_id = response['files'][0]['id']
                logger.info(f"Found existing Google Drive folder: {self.folder_name} (ID: {folder_id})")
                return folder_id
            
            # Create folder if it doesn't exist
            folder_metadata = {
                'name': self.folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            file = self.drive_service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            folder_id = file.get('id')
            logger.info(f"Created Google Drive folder: {self.folder_name} (ID: {folder_id})")
            return folder_id
            
        except Exception as e:
            logger.error(f"Error getting/creating Google Drive folder: {e}")
            return None
    
    def save_trade_data(self, trade):
        """
        Save trade data to CSV and upload to Google Drive
        
        Args:
            trade (dict): Trade data
        """
        local_path = os.path.join(self.local_data_dir, self.trades_file)
        
        try:
            # Flatten nested dictionaries
            flat_trade = self._flatten_dict(trade)
            
            # Convert to DataFrame
            trade_df = pd.DataFrame([flat_trade])
            
            # Append or create file
            if os.path.exists(local_path):
                existing_df = pd.read_csv(local_path)
                combined_df = pd.concat([existing_df, trade_df], ignore_index=True)
                combined_df.to_csv(local_path, index=False)
            else:
                trade_df.to_csv(local_path, index=False)
            
            logger.info(f"Saved trade data to {local_path}")
            
            # Upload to Google Drive
            self._upload_file(local_path, self.trades_file)
            
        except Exception as e:
            logger.error(f"Error saving trade data: {e}")
    
    def save_portfolio_value(self, value):
        """
        Save portfolio value to CSV and upload to Google Drive
        
        Args:
            value (float): Portfolio value
        """
        local_path = os.path.join(self.local_data_dir, self.portfolio_file)
        
        try:
            # Create portfolio data
            portfolio_data = {
                'timestamp': datetime.now().isoformat(),
                'value': value
            }
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame([portfolio_data])
            
            # Append or create file
            if os.path.exists(local_path):
                existing_df = pd.read_csv(local_path)
                combined_df = pd.concat([existing_df, portfolio_df], ignore_index=True)
                combined_df.to_csv(local_path, index=False)
            else:
                portfolio_df.to_csv(local_path, index=False)
            
            logger.info(f"Saved portfolio value to {local_path}")
            
            # Upload to Google Drive
            self._upload_file(local_path, self.portfolio_file)
            
        except Exception as e:
            logger.error(f"Error saving portfolio value: {e}")
    
    def load_trades(self):
        """
        Load trade data from CSV
        
        Returns:
            pandas.DataFrame: Trade data or empty DataFrame if file doesn't exist
        """
        local_path = os.path.join(self.local_data_dir, self.trades_file)
        
        try:
            # Download from Google Drive
            self._download_file(self.trades_file, local_path)
            
            # Load data
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading trade data: {e}")
            return pd.DataFrame()
    
    def load_portfolio_value(self):
        """
        Load latest portfolio value
        
        Returns:
            dict: Portfolio data or None if file doesn't exist
        """
        local_path = os.path.join(self.local_data_dir, self.portfolio_file)
        
        try:
            # Download from Google Drive
            self._download_file(self.portfolio_file, local_path)
            
            # Load data
            if os.path.exists(local_path):
                df = pd.read_csv(local_path)
                if not df.empty:
                    latest = df.iloc[-1].to_dict()
                    return latest
            
            return None
                
        except Exception as e:
            logger.error(f"Error loading portfolio value: {e}")
            return None
    
    def _upload_file(self, local_path, filename):
        """
        Upload file to Google Drive
        
        Args:
            local_path (str): Local file path
            filename (str): File name in Google Drive
        """
        if not self.drive_service or not self.folder_id:
            logger.warning("Google Drive service not available. Skipping upload.")
            return
            
        try:
            # Check if file exists
            query = f"name='{filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(q=query).execute()
            
            file_metadata = {
                'name': filename,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(local_path)
            
            if response.get('files'):
                # Update existing file
                file_id = response['files'][0]['id']
                self.drive_service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                logger.info(f"Updated file in Google Drive: {filename}")
            else:
                # Create new file
                self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"Uploaded file to Google Drive: {filename}")
                
        except Exception as e:
            logger.error(f"Error uploading file to Google Drive: {e}")
    
    def _download_file(self, filename, local_path):
        """
        Download file from Google Drive
        
        Args:
            filename (str): File name in Google Drive
            local_path (str): Local file path
        """
        if not self.drive_service or not self.folder_id:
            logger.warning("Google Drive service not available. Skipping download.")
            return
            
        try:
            # Check if file exists
            query = f"name='{filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(q=query).execute()
            
            if not response.get('files'):
                logger.warning(f"File not found in Google Drive: {filename}")
                return
                
            file_id = response['files'][0]['id']
            
            request = self.drive_service.files().get_media(fileId=file_id)
            
            with io.BytesIO() as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                
                # Save to local file
                with open(local_path, 'wb') as f:
                    f.write(fh.getvalue())
            
            logger.info(f"Downloaded file from Google Drive: {filename}")
                
        except Exception as e:
            logger.error(f"Error downloading file from Google Drive: {e}")
    
    def historical_data_exists(self):
        """
        Check if historical trade data exists
        
        Returns:
            bool: True if data exists
        """
        trades = self.load_trades()
        return not trades.empty
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten nested dictionary
        
        Args:
            d (dict): Dictionary to flatten
            parent_key (str): Parent key
            sep (str): Separator
            
        Returns:
            dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)
