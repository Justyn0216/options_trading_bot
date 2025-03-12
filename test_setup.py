#!/usr/bin/env python
"""
Test script to validate the setup of the options trading bot
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check that environment variables are set correctly"""
    load_dotenv()
    
    required_vars = [
        'TRADIER_API_KEY',
        'TRADIER_ACCOUNT_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False
    
    logger.info("Environment variables check: PASSED")
    return True

def check_imports():
    """Check that all required packages are installed"""
    required_packages = [
        'requests',
        'pandas',
        'numpy',
        'scipy',
        'sklearn',
        'joblib',
        'google',
        'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install these packages using: pip install -r requirements.txt")
        return False
    
    logger.info("Package imports check: PASSED")
    return True

def check_google_credentials():
    """Check that Google credentials file exists"""
    credentials_file = os.getenv('GOOGLE_CREDENTIALS_FILE', 'google_credentials.json')
    
    if not os.path.exists(credentials_file):
        logger.error(f"Google credentials file not found: {credentials_file}")
        logger.error("Please download your service account credentials from Google Cloud Console")
        return False
    
    logger.info("Google credentials check: PASSED")
    return True

def check_project_structure():
    """Check that the project structure is correct"""
    required_dirs = [
        'options_trading_bot',
        'options_trading_bot/models',
        'options_trading_bot/apis',
        'options_trading_bot/scanner',
        'options_trading_bot/trading',
        'options_trading_bot/notifications',
        'options_trading_bot/data',
        'options_trading_bot/ml'
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        logger.error(f"Missing required directories: {', '.join(missing_dirs)}")
        logger.error("Please create these directories according to the project structure")
        return False
    
    logger.info("Project structure check: PASSED")
    return True

def main():
    """Run all checks"""
    logger.info("Starting setup validation...")
    
    checks = [
        check_environment,
        check_imports,
        check_google_credentials,
        check_project_structure
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    if all_passed:
        logger.info("All checks passed! The options trading bot is ready to run.")
        logger.info("You can start the bot using: python -m options_trading_bot.main")
        return 0
    else:
        logger.error("Some checks failed. Please fix the issues and run this script again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
