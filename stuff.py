# 1. First, let's completely replace the analyze_ticker_before_scan method in order_flow_integration.py

def analyze_ticker_before_scan(self, ticker):
    """
    Analyze a ticker's order flow before scanning
    
    Parameters:
    ticker: Ticker symbol
    
    Returns:
    dict: Order flow analysis results
    """
    # IMPORTANT: Skip if automatic analysis is disabled
    if hasattr(self.trading_bot, 'skip_automatic_order_flow') and self.trading_bot.skip_automatic_order_flow:
        logger.info(f"Skipping automatic order flow analysis for {ticker} - using sequential analysis instead")
        return {'symbol': ticker, 'proceed_with_scan': True, 'skip_analysis': True}
    
    # Skip if already analyzed in this scan cycle
    if hasattr(self.trading_bot, 'order_flow_already_analyzed') and ticker in self.trading_bot.order_flow_already_analyzed:
        logger.info(f"Skipping duplicate order flow analysis for {ticker} - already analyzed in this scan")
        return {'symbol': ticker, 'proceed_with_scan': True, 'skip_analysis': True}
        
    try:
        # Check if order flow analysis for this ticker was recently completed
        now = time.time()
    
        # Use the main analyzer's timestamps if available
        if hasattr(self.order_flow_analyzer, 'last_update_time') and ticker in self.order_flow_analyzer.last_update_time:
            last_analysis = self.order_flow_analyzer.last_update_time[ticker]
            if now - last_analysis < 300:  # 5 minutes
                logger.info(f"Skipping duplicate order flow analysis for {ticker} - recently analyzed by main analyzer")
                return {'symbol': ticker, 'proceed_with_scan': True, 'skip_analysis': True}
        
        # Check If Already Analyzed by integrator
        if not hasattr(self, 'last_ticker_analysis_time'):
            self.last_ticker_analysis_time = {}
        
        # Skip if analyzed in the last 5 minutes
        if ticker in self.last_ticker_analysis_time:
            if now - self.last_ticker_analysis_time[ticker] < 300:  # 5 minutes
                logger.info(f"Skipping order flow analysis for {ticker} - analyzed recently")
                return {'symbol': ticker, 'proceed_with_scan': True, 'skip_analysis': True}
            
        # Record this analysis time
        self.last_ticker_analysis_time[ticker] = now

        logger.info(f"Analyzing order flow for {ticker} before scan")
        
        # Calculate put/call ratio - COMPLETE THE STEPS SEQUENTIALLY
        logger.info(f"Calculating put/call ratio for {ticker}")
        put_call_ratio = self.order_flow_analyzer.calculate_put_call_ratio(ticker)
        
        # Generate a brief report
        report = {
            'symbol': ticker,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'put_call_ratio': put_call_ratio,
            'sentiment': put_call_ratio.get('sentiment', 'neutral'),
            'proceed_with_scan': True
        }
        
        # Log the main metrics
        logger.info(f"Order flow for {ticker}: P/C Ratio={put_call_ratio.get('volume_put_call_ratio', 1.0):.2f}, Sentiment={put_call_ratio.get('sentiment', 'neutral')}")
        
        # Optionally, we may decide not to scan if put/call ratio indicates extreme sentiment
        # This would be configurable via the config
        extreme_ratio_threshold = self.config.get('order_flow', {}).get('extreme_ratio_threshold', 5.0)
        
        if put_call_ratio.get('volume_put_call_ratio', 1.0) > extreme_ratio_threshold:
            report['proceed_with_scan'] = False
            logger.warning(f"Skipping scan for {ticker} due to extreme put/call ratio: {put_call_ratio.get('volume_put_call_ratio', 1.0):.2f}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error analyzing ticker before scan: {e}", exc_info=True)
        return {
            'symbol': ticker,
            'proceed_with_scan': True,
            'error': str(e)
        }

# 2. Next, let's create a patch for option_scanner.py - add this method

def scan_for_sequential_execution(self, ticker_list):
    """
    Special scanner method that skips all order flow integration for sequential execution
    
    Parameters:
    ticker_list: List of tickers to scan
    
    Returns:
    list: List of undervalued options
    """
    try:
        # Default ticker list if none provided
        if ticker_list is None:
            ticker_list = ['SPY', 'QQQ', 'IWM']
        
        all_undervalued_options = []
        ticker_results = {}
        
        # Store original symbol
        original_symbol = self.symbol
        
        # Scan each ticker
        for ticker in ticker_list:
            try:
                # Update the scanner symbol
                self.symbol = ticker
                
                # Get current stock price
                quote = self.tradier_api.get_quote(ticker)
                
                if not quote:
                    logger.error(f"Could not get quote for {ticker}")
                    continue
                
                stock_price = quote['last']
                
                # Define strike price limitations to avoid far OTM options
                max_call_strike = stock_price * 1.10  # 10% above current price
                min_put_strike = stock_price * 0.90   # 10% below current price
                
                # Improved minimum liquidity requirements
                min_volume = 50
                min_open_interest = 100
                min_bid = 0.05  # Minimum bid price to avoid penny options
                
                # Get option expirations
                expirations = self.tradier_api.get_option_expirations(ticker)
                
                if not expirations:
                    logger.warning(f"No option expirations found for {ticker}")
                    continue
                
                # Filter expirations by DTE
                filtered_expirations = []
                for exp in expirations:
                    # Handle different expiration formats
                    if isinstance(exp, dict) and 'date' in exp:
                        expiry_date = exp['date']
                    elif isinstance(exp, str):
                        expiry_date = exp
                    else:
                        logger.warning(f"Unexpected expiration format: {type(exp)}")
                        continue
                        
                    try:
                        dte = self.get_days_to_expiration(expiry_date)
                        
                        if 1 <= dte <= 90:  # Default min_dte=1, max_dte=90
                            filtered_expirations.append(expiry_date)
                    except Exception as exp_error:
                        logger.warning(f"Error processing expiration {exp}: {exp_error}")
                        continue
                
                # Get all options for filtered expirations
                undervalued_options = []
                
                for expiry in filtered_expirations:
                    try:
                        option_chain = self.tradier_api.get_option_chain(ticker, expiry)
                        
                        if not option_chain:
                            logger.warning(f"Empty option chain for {ticker} expiry {expiry}")
                            continue
                        
                        # Validate the option chain is a list
                        if not isinstance(option_chain, list):
                            logger.warning(f"Expected option_chain to be a list, got {type(option_chain)}")
                            if isinstance(option_chain, dict):
                                # Sometimes API might return a single option as dict instead of list
                                option_chain = [option_chain]
                            else:
                                continue
                        
                        for option in option_chain:
                            try:
                                # Ensure option is a dictionary
                                if not isinstance(option, dict):
                                    logger.warning(f"Expected option to be a dict, got {type(option)}")
                                    continue
                                    
                                # Skip options with insufficient liquidity
                                if option.get('volume', 0) < min_volume or option.get('open_interest', 0) < min_open_interest:
                                    continue
                                
                                # Skip options with very low bid prices
                                if option.get('bid', 0) < min_bid:
                                    continue
                                
                                # Skip options that are too far OTM 
                                option_type = option.get('option_type', '').lower()
                                strike = option.get('strike', 0)
                                
                                if option_type == 'call' and strike > max_call_strike:
                                    continue  # Skip far out-of-the-money calls
                                elif option_type == 'put' and strike < min_put_strike:
                                    continue  # Skip far out-of-the-money puts
                                
                                # Skip options with wide bid-ask spread
                                bid = option.get('bid', 0)
                                ask = option.get('ask', 0)
                                if bid > 0 and ask > 0:
                                    mid = (bid + ask) / 2
                                    if mid > 0:
                                        spread_pct = (ask - bid) / mid
                                        if spread_pct > 0.20:  # 20% max spread
                                            continue
                                
                                # Evaluate the option
                                evaluation = self.evaluate_option(option, stock_price)
                                
                                if evaluation and evaluation['is_undervalued']:
                                    # Additional filter: Require decent delta for directional exposure
                                    delta_abs = abs(evaluation.get('delta', 0))
                                    if delta_abs > 0.20 and delta_abs < 0.80:  # Not too deep ITM or OTM
                                        undervalued_options.append(evaluation)
                            
                            except Exception as option_error:
                                logger.error(f"Error processing option: {option_error}")
                                continue
                    
                    except Exception as exp_error:
                        logger.error(f"Error processing expiration {expiry}: {exp_error}")
                        continue
                
                # Sort by percentage difference (most undervalued first)
                if undervalued_options:
                    undervalued_options.sort(key=lambda x: x.get('diff_percent', 0), reverse=True)
                
                all_undervalued_options.extend(undervalued_options)
                ticker_results[ticker] = {
                    'options_scanned': 0,  # We're not tracking this in this version
                    'undervalued_found': len(undervalued_options)
                }
                
                # Log results
                logger.info(f"Scanned {ticker}: Found {len(undervalued_options)} undervalued options")
                
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue
        
        # Restore original symbol
        self.symbol = original_symbol
        
        # Sort all options by percentage difference (most undervalued first)
        if all_undervalued_options:
            all_undervalued_options.sort(key=lambda x: x.get('diff_percent', 0), reverse=True)
        
        logger.info(f"Found {len(all_undervalued_options)} undervalued options across all tickers")
        return all_undervalued_options
        
    except Exception as e:
        logger.error(f"Error in scan_for_sequential_execution: {e}")
        return []

# 3. Now let's update the scan_and_trade method in integrated_bot.py

def scan_and_trade(self):
    """
    Perform a single scan and potentially enter/exit trades with strictly sequential execution
    """
    logger.info(f"Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Track scan cycles with unique ID
    if not hasattr(self, 'current_scan_id'):
        self.current_scan_id = 0
    self.current_scan_id += 1
    current_scan = self.current_scan_id
    logger.info(f"Starting scan cycle #{current_scan}")

    # Flags to disable automatic order flow analysis during the scan
    self.skip_automatic_order_flow = True
    self.order_flow_already_analyzed = {}  # Track which tickers were analyzed
    
    # Step 1: Assess portfolio risk
    logger.info("Step 1: Assessing portfolio risk")
    risk_assessment = self.portfolio_risk_manager.assess_portfolio_risk(
        self.trading_system.active_trades,
        self.trading_system.portfolio_value
    )
    
    # Log risk level
    logger.info(f"Current risk level: {risk_assessment['risk_level']} ({risk_assessment['risk_score']:.1f}/100)")
    
    # Step 2: Check existing trades for exit signals
    logger.info("Step 2: Checking existing trades for exit signals")
    exited_trades = []
    
    try:
        for i in range(len(self.trading_system.active_trades) - 1, -1, -1):
            trade = self.trading_system.active_trades[i]
            
            # Get current quote for this option
            quote = self.tradier_api.get_quote(trade['symbol'])
            
            if not quote:
                continue
            
            current_price = quote.get('last', quote.get('ask'))
            
            # Get current market data for exit evaluation
            current_market_data = {
                'last': current_price,
                'bid': quote.get('bid', 0),
                'ask': quote.get('ask', 0),
                'mid_price': (quote.get('bid', 0) + quote.get('ask', 0)) / 2
            }
            
            # Get option data for Greeks and IV
            option_chain = self.tradier_api.get_option_chain(trade['underlying'], trade['expiration'])
            option_data = next((opt for opt in option_chain if opt['symbol'] == trade['symbol']), None)
            
            if option_data:
                # Update market data with Greeks and IV
                if 'greeks' in option_data:
                    greeks = option_data['greeks']
                    current_market_data.update({
                        'delta': greeks.get('delta', trade.get('delta', 0)),
                        'gamma': greeks.get('gamma', trade.get('gamma', 0)),
                        'theta': greeks.get('theta', trade.get('theta', 0)),
                        'vega': greeks.get('vega', trade.get('vega', 0)),
                        'rho': greeks.get('rho', trade.get('rho', 0)),
                        'iv': greeks.get('mid_iv', trade.get('iv', 0))
                    })
                
                # Get model price for valuation
                underlying_quote = self.tradier_api.get_quote(trade['underlying'])
                
                if underlying_quote:
                    stock_price = underlying_quote['last']
                    evaluation = self.option_scanner.evaluate_option(option_data, stock_price)
                    
                    if evaluation:
                        current_market_data.update({
                            'model_price': evaluation['weighted_price'],
                            'status': evaluation['status'],
                            'diff_percent': evaluation['diff_percent']
                        })
            
            # Get the price history, ensure it's a list
            price_history = trade.get('price_history', [])
            if not isinstance(price_history, list):
                price_history = []

            # Then append the current price
            if 'current_price' in trade:
                price_history.append(trade['current_price'])
            
            # Get exit recommendation from advanced exit strategies
            exit_recommendation = self.exit_strategies.get_exit_recommendation(
                trade, current_market_data, price_history
            )
            
            # Update trade with current price history
            trade['price_history'] = price_history
            
            # Update trade with current data
            if 'current_price' not in trade or trade['current_price'] != current_price:
                trade['current_price'] = current_price
                # Also update high price for trailing stop purposes
                if 'high_price' not in trade or current_price > trade['high_price']:
                    trade['high_price'] = current_price
            
            # Check if we should exit this trade
            if exit_recommendation['should_exit']:
                logger.info(f"Exiting trade {trade['symbol']} due to: {exit_recommendation['exit_reason']}")
                exited_trade = self.trading_system.exit_trade(i, current_price, exit_recommendation['exit_reason'])
                if exited_trade:
                    exited_trades.append(exited_trade)
                    
                    # Record the trade exit time for cooldown mechanism
                    self.recently_exited_trades[exited_trade['symbol']] = time.time()
                    
                    # Log detailed exit reason
                    logger.info(f"Exit details: {', '.join(exit_recommendation['explanation'])}")
                    
                    # IMPORTANT: Immediately sync with Google Drive after each trade exit
                    # This ensures the trade is recorded even if there's an error later
                    self.sync_with_google_drive()
            else:
                # If not exiting, update trade status
                updated_trade, status_changed, new_status = self.trading_system.update_trade(
                    i, current_price, current_market_data.get('model_price', 0)
                )
                
                # Send notification if status changed
                if status_changed and self.notification_manager:
                    self.notification_manager.notify_status_change(updated_trade, new_status)
        
    except Exception as e:
        logger.error(f"Error during trade exit checks: {e}", exc_info=True)
        # Make sure to sync with Google Drive even if there's an error
        self.sync_with_google_drive()
    
    # Step 3: Re-train ML model after exiting trades
    if exited_trades:
        logger.info(f"Step 3: Exited {len(exited_trades)} trades, retraining ML model")
        self.ml_predictor.train_models()
        
        # Generate performance plots
        self.ml_predictor.plot_performance_over_time()
        
        # Make sure trade data is synced to Google Drive after exits
        self.sync_with_google_drive()
    else:
        logger.info("Step 3: No trades exited, skipping ML retraining")
    
    # Step 4: Get ticker list from config or use default
    ticker_list = self.config.get('trading', {}).get('scan_tickers', None)
    logger.info(f"Step 4: Got ticker list with {len(ticker_list) if ticker_list else 0} tickers")
    
    # Step 5: Process each ticker SEQUENTIALLY - running all analysis types for each ticker
    ticker_results = {}
    
    # Limit to 5 tickers for performance reasons
    tickers_to_process = ticker_list[:5] if ticker_list else ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
    
    # Process tickers one at a time
    for ticker in tickers_to_process:
        logger.info(f"===== Processing ticker {ticker} sequentially =====")
        
        # 1. Time & sales analysis for this ticker
        logger.info(f"Running time & sales analysis for {ticker}")
        ts_features = {}
        if hasattr(self, 'time_sales_analyzer'):
            try:
                ts_features = self.time_sales_analyzer.get_time_sales_features(ticker)
                logger.info(f"Completed time & sales analysis for {ticker}")
            except Exception as e:
                logger.error(f"Error in time sales analysis for {ticker}: {e}")
        
        # 2. Order flow analysis for this ticker
        logger.info(f"Running order flow analysis for {ticker}")
        of_data = {}
        if hasattr(self, 'order_flow_analyzer'):
            try:
                # IMPORTANT: Mark this ticker as analyzed to prevent duplicate analysis
                self.order_flow_already_analyzed[ticker] = True
                
                # IMPORTANT: Complete each operation fully before starting the next
                logger.info(f"Taking option chain snapshot for {ticker}")
                snapshot = self.order_flow_analyzer.take_option_chain_snapshot(ticker)
                
                logger.info(f"Calculating put/call ratio for {ticker}")
                pc_ratio = self.order_flow_analyzer.calculate_put_call_ratio(ticker)
                
                logger.info(f"Detecting volume spikes for {ticker}")
                unusual_activity = self.order_flow_analyzer.detect_volume_spikes(ticker)
                
                of_data = {
                    'put_call_ratio': pc_ratio,
                    'unusual_activity_count': len(unusual_activity),
                    'unusual_activity': unusual_activity[:5]
                }
                logger.info(f"Completed all order flow analysis for {ticker}")
            except Exception as e:
                logger.error(f"Error in order flow analysis for {ticker}: {e}")
        
        # 3. Option scanning for this ticker
        logger.info(f"Scanning options for {ticker}")
        ticker_options = []
        try:
            # IMPORTANT: Use our special version that skips order flow
            if hasattr(self.option_scanner, 'scan_for_sequential_execution'):
                ticker_options = self.option_scanner.scan_for_sequential_execution([ticker])
            else:
                # Add the method if it doesn't exist
                from types import MethodType
                self.option_scanner.scan_for_sequential_execution = MethodType(
                    scan_for_sequential_execution, self.option_scanner
                )
                ticker_options = self.option_scanner.scan_for_sequential_execution([ticker])
                
            logger.info(f"Found {len(ticker_options)} undervalued options for {ticker}")
        except Exception as e:
            logger.error(f"Error scanning options for {ticker}: {e}")
        
        # Store results for this ticker
        ticker_results[ticker] = {
            'time_sales': ts_features,
            'order_flow': of_data,
            'options': ticker_options
        }
        
        # Brief pause between tickers to prevent API rate limiting
        time.sleep(1)
        logger.info(f"===== Completed all analysis for {ticker} =====")
    
    # Step 6: Process all results for trade decisions
    logger.info("Step 6: Processing all results for trade decisions")
    
    # Gather all undervalued options across all tickers
    all_options = []
    for ticker, results in ticker_results.items():
        if results['options']:
            all_options.extend(results['options'])
    
    logger.info(f"Found {len(all_options)} undervalued options across all tickers")
    
    # Apply novelty filter to prioritize diverse options
    logger.info("Applying novelty filter to prioritize diverse trade options")
    prioritized_options = self.novelty_filter.prioritize_options(all_options)
    
    if prioritized_options:
        logger.info(f"After prioritization, top option has score: {prioritized_options[0]['combined_score']:.2f}")
        
        # Log details about top options for debugging
        for idx, opt in enumerate(prioritized_options[:3]):  # Log top 3 options
            logger.info(f"Top option {idx+1}: {opt['symbol']} - {opt['diff_percent']:.2f}% undervalued, Strike: {opt['strike']}, Exp: {opt['expiration']}, Type: {opt['option_type']}")
    else:
        logger.info("No viable options found after filtering")
        self.sync_with_google_drive()
        logger.info(f"Completed scan cycle #{current_scan}")
        return
    
    # Step 7: Make trade decisions based on risk assessment
    logger.info("Step 7: Making trade decisions based on risk assessment")
    should_enter_new_trades = risk_assessment['risk_level'] != 'VERY HIGH'
    
    if should_enter_new_trades and len(self.trading_system.active_trades) < 5:  # Limit to 5 concurrent trades
        logger.info("Portfolio risk level acceptable for new trades")
        
        # Enhance options with time & sales and order flow data if available
        enhanced_options = []
        for option in prioritized_options:
            # Start with the original option
            enhanced_option = option.copy()
            underlying = enhanced_option.get('underlying', '')
            
            # Add time & sales data if available
            if underlying in ticker_results and 'time_sales' in ticker_results[underlying] and hasattr(self, 'time_sales_analyzer'):
                try:
                    enhanced_option = self.time_sales_analyzer.evaluate_option_with_time_sales(
                        enhanced_option, ticker_results[underlying]['time_sales'])
                except Exception as e:
                    logger.error(f"Error enhancing option with time & sales: {e}")
            
            # Add order flow data if available
            if underlying in ticker_results and 'order_flow' in ticker_results[underlying] and hasattr(self, 'order_flow_integrator'):
                try:
                    enhanced_option = self.order_flow_integrator.enhance_option_with_order_flow(enhanced_option)
                except Exception as e:
                    logger.error(f"Error enhancing option with order flow: {e}")
            
            enhanced_options.append(enhanced_option)
        
        # Process the enhanced options for trade decisions
        for option in enhanced_options:
            # Check if this option is in cooldown period (recently traded)
            if option['symbol'] in self.recently_exited_trades:
                last_exit_time = self.recently_exited_trades[option['symbol']]
                cooldown_period = 3600  # 1 hour in seconds
                if time.time() - last_exit_time < cooldown_period:
                    logger.info(f"Skipping {option['symbol']} - still in cooldown period")
                    continue
            
            # Log novelty score
            logger.info(f"Considering {option['symbol']} ({option['underlying']}) - Undervalued: {option['diff_percent']:.2f}%, " +
                      f"Novelty: {option.get('novelty_score', 0):.2f}, Combined: {option.get('combined_score', 0):.2f}")
            
            # Check if this trade should be allowed given portfolio risk
            allow_trade, reason, _ = self.portfolio_risk_manager.should_allow_new_trade(
                option, 
                self.trading_system.portfolio_value,
                self.trading_system.active_trades
            )
            
            if not allow_trade:
                logger.info(f"Skipping trade {option['symbol']} due to risk constraint: {reason}")
                continue
            
            # Log detailed info before ML prediction
            logger.info(f"Evaluating ML prediction for {option['symbol']} - Delta: {option.get('delta', 'N/A')}, IV: {option.get('iv', 'N/A')}")
            
            # Use ML to predict if we should enter this trade
            should_enter, prediction_results = self.ml_predictor.should_enter_trade(option)
            
            # Add detailed logging after ML prediction
            if should_enter:
                logger.info(f"ML approved trade for {option['symbol']} with confidence: {prediction_results.get('confidence', 'N/A')}")
            else:
                logger.info(f"ML rejected trade for {option['symbol']} - Confidence: {prediction_results.get('confidence', 'N/A')}, Probability: {prediction_results.get('success_probability', 0):.2f}")
            
            if should_enter:
                # Get position sizing recommendation
                position_sizing = self.portfolio_risk_manager.generate_position_sizing_recommendation(
                    option,
                    self.trading_system.portfolio_value,
                    self.trading_system.active_trades
                )
                
                # Override position size with risk-based recommendation
                option['position_size_override'] = position_sizing['recommended_contracts']
                
                # Enter trade
                new_trade = self.trading_system.enter_trade(option)
                
                if new_trade:
                    logger.info(
                        f"Entered new trade: {new_trade['symbol']} with ML confidence: "
                        f"{prediction_results.get('confidence', 'N/A')} ({prediction_results.get('success_probability', 0):.2f})"
                    )
                    
                    # Generate a risk report after entering a new trade
                    self.portfolio_risk_manager.generate_risk_report(
                        self.trading_system.active_trades,
                        self.trading_system.portfolio_value
                    )
                    
                    # Sync with Google Drive after entering a new trade
                    self.sync_with_google_drive()
                    
                    # Only enter one trade per scan
                    break
    else:
        if not should_enter_new_trades:
            logger.info("Step 7: Skipping new trades due to high portfolio risk")
        elif len(self.trading_system.active_trades) >= 5:
            logger.info("Step 7: Maximum number of concurrent trades reached")
    
    # Step 8: Reset state for next cycle and sync with Google Drive
    logger.info("Step 8: Resetting analysis state and syncing with Google Drive")
    
    # Clear the analysis flags and caches to ensure fresh analysis next cycle
    if hasattr(self, 'order_flow_integrator') and hasattr(self.order_flow_integrator, 'last_ticker_analysis_time'):
        # Clear the analysis timestamps to force fresh analysis next cycle
        self.order_flow_integrator.last_ticker_analysis_time = {}
    
    # Also clear local tracking if it exists
    if hasattr(self, 'last_ticker_scan_time'):
        self.last_ticker_scan_time = {}
    
    # Clear the order flow analysis flags
    self.skip_automatic_order_flow = False
    self.order_flow_already_analyzed = {}

    # Sync with Google Drive
    self.sync_with_google_drive()
    
    logger.info(f"Completed scan cycle #{current_scan}")
