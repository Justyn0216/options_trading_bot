# options_trading_bot/models/option_pricing.py

import numpy as np
import scipy.stats as si
from scipy.optimize import fsolve
import logging

logger = logging.getLogger(__name__)

class OptionPricingEngine:
    """
    Implements multiple option pricing models:
    1. Black-Scholes
    2. Merton Jump Diffusion
    3. Barone-Adesi Whaley (for American options)
    4. Monte Carlo simulation
    5. Binomial Tree model
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Default risk-free rate (2%)
        self.dividend_yield = 0.0   # Default dividend yield
    
    def update_market_params(self, risk_free_rate=None, dividend_yield=None):
        """Update market parameters if provided"""
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        if dividend_yield is not None:
            self.dividend_yield = dividend_yield
    
    def calculate_weighted_price(self, option_data, weights=None):
        """
        Calculate weighted average price using all models
        
        Args:
            option_data (dict): Option data containing required parameters
            weights (dict): Model weights, if None, equal weights are used
            
        Returns:
            float: Weighted average theoretical price
        """
        if weights is None:
            # Equal weights if not specified
            weights = {
                'black_scholes': 0.2,
                'merton_jump': 0.2,
                'barone_adesi': 0.2,
                'monte_carlo': 0.2,
                'binomial': 0.2
            }
        
        # Extract option parameters
        S = option_data['underlying_price']
        K = option_data['strike']
        T = option_data['time_to_expiry']  # in years
        sigma = option_data['implied_volatility']
        option_type = option_data['option_type'].lower()  # 'call' or 'put'
        is_american = option_data.get('is_american', True)  # Default to American options
        
        # Calculate prices using each model
        try:
            prices = {
                'black_scholes': self.black_scholes(S, K, T, sigma, option_type),
                'merton_jump': self.merton_jump(S, K, T, sigma, option_type),
                'monte_carlo': self.monte_carlo(S, K, T, sigma, option_type),
                'binomial': self.binomial_tree(S, K, T, sigma, option_type, is_american)
            }
            
            # Barone-Adesi only for American options
            if is_american:
                prices['barone_adesi'] = self.barone_adesi_whaley(S, K, T, sigma, option_type)
            else:
                # For European options, use Black-Scholes instead
                prices['barone_adesi'] = prices['black_scholes']
                
            # Calculate weighted average price
            weighted_price = sum(prices[model] * weights[model] for model in prices)
            
            # Log individual model prices
            logger.debug(f"Model prices for {option_data['symbol']}: {prices}")
            logger.debug(f"Weighted price: {weighted_price:.4f}")
            
            return weighted_price
            
        except Exception as e:
            logger.error(f"Error calculating weighted price: {e}")
            # If calculation fails, return None
            return None
    
    def black_scholes(self, S, K, T, sigma, option_type):
        """
        Black-Scholes option pricing model
        
        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price
            T (float): Time to expiration in years
            sigma (float): Implied volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Option price
        """
        if sigma <= 0 or T <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = (S * np.exp(-self.dividend_yield * T) * si.norm.cdf(d1, 0.0, 1.0) - 
                     K * np.exp(-self.risk_free_rate * T) * si.norm.cdf(d2, 0.0, 1.0))
        else:  # put option
            price = (K * np.exp(-self.risk_free_rate * T) * si.norm.cdf(-d2, 0.0, 1.0) - 
                     S * np.exp(-self.dividend_yield * T) * si.norm.cdf(-d1, 0.0, 1.0))
        
        return price
    
    def merton_jump(self, S, K, T, sigma, option_type, lambda_=1.0, jump_mean=0.0, jump_std=0.2):
        """
        Merton Jump Diffusion model
        
        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price
            T (float): Time to expiration in years
            sigma (float): Implied volatility
            option_type (str): 'call' or 'put'
            lambda_ (float): Jump intensity (average number of jumps per year)
            jump_mean (float): Average jump size
            jump_std (float): Standard deviation of jump size
            
        Returns:
            float: Option price
        """
        if sigma <= 0 or T <= 0:
            return 0.0
        
        # Maximum number of jumps to consider
        M = 20
        price = 0.0
        
        # Sum over possible number of jumps
        for k in range(M):
            # Probability of k jumps occurring within time T
            p_k = np.exp(-lambda_ * T) * (lambda_ * T) ** k / np.math.factorial(k)
            
            # Adjusted volatility incorporating jump component
            sigma_k = np.sqrt(sigma ** 2 + k * jump_std ** 2 / T)
            
            # Adjusted mean incorporating jump component
            mu_k = self.risk_free_rate - self.dividend_yield - lambda_ * (np.exp(jump_mean + 0.5 * jump_std ** 2) - 1) + k * jump_mean / T
            
            # Black-Scholes price with adjusted parameters
            d1 = (np.log(S / K) + (mu_k + 0.5 * sigma_k ** 2) * T) / (sigma_k * np.sqrt(T))
            d2 = d1 - sigma_k * np.sqrt(T)
            
            if option_type == 'call':
                bs_price = (S * np.exp((mu_k - self.risk_free_rate + self.dividend_yield) * T) * si.norm.cdf(d1, 0.0, 1.0) - 
                         K * np.exp(-self.risk_free_rate * T) * si.norm.cdf(d2, 0.0, 1.0))
            else:  # put option
                bs_price = (K * np.exp(-self.risk_free_rate * T) * si.norm.cdf(-d2, 0.0, 1.0) - 
                         S * np.exp((mu_k - self.risk_free_rate + self.dividend_yield) * T) * si.norm.cdf(-d1, 0.0, 1.0))
            
            price += p_k * bs_price
        
        return price
        
    def barone_adesi_whaley(self, S, K, T, sigma, option_type):
        """
        Barone-Adesi and Whaley approximation for American options
        
        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price
            T (float): Time to expiration in years
            sigma (float): Implied volatility
            option_type (str): 'call' or 'put'
            
        Returns:
            float: Option price
        """
        if sigma <= 0 or T <= 0:
            return 0.0
        
        # For very short-dated options, use binomial model instead
        if T < 0.01:
            return self.binomial_tree(S, K, T, sigma, option_type, True)
        
        # European option price using Black-Scholes
        european_price = self.black_scholes(S, K, T, sigma, option_type)
        
        # Parameters for the approximation
        r = self.risk_free_rate
        q = self.dividend_yield
        b = r - q  # Cost of carry
        
        M = 2 * r / (sigma ** 2)
        N = 2 * b / (sigma ** 2)
        
        if option_type == 'call':
            # For American calls with no dividends, use European price
            if q <= 0:
                return european_price
            
            # Critical price calculation
            q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
            S_critical = K / (1 - 1 / q2) if q2 > 0 else float('inf')
            
            # Early exercise premium
            if S < S_critical:
                A2 = (S / q2) * (1 - np.exp((b - r) * T) * si.norm.cdf((np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))))
                return european_price + A2
            else:
                return S - K  # Immediate exercise
        
        else:  # put option
            # Critical price calculation
            q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
            S_critical = K * (1 - 1 / q1) / (1 - np.exp((b - r) * T))
            
            # Early exercise premium
            if S > S_critical:
                A1 = -(S / q1) * (1 - np.exp((b - r) * T) * si.norm.cdf(-(np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))))
                return european_price + A1
            else:
                return K - S  # Immediate exercise
    
    def monte_carlo(self, S, K, T, sigma, option_type, num_simulations=10000, num_steps=100):
        """
        Monte Carlo simulation for option pricing
        
        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price
            T (float): Time to expiration in years
            sigma (float): Implied volatility
            option_type (str): 'call' or 'put'
            num_simulations (int): Number of price path simulations
            num_steps (int): Number of time steps in each simulation
            
        Returns:
            float: Option price
        """
        if sigma <= 0 or T <= 0:
            return 0.0
        
        dt = T / num_steps
        drift = (self.risk_free_rate - self.dividend_yield - 0.5 * sigma ** 2) * dt
        volatility = sigma * np.sqrt(dt)
        
        # Initialize array for final stock prices
        final_prices = np.zeros(num_simulations)
        
        # Simulate stock price paths
        for i in range(num_simulations):
            price_path = np.zeros(num_steps + 1)
            price_path[0] = S
            
            for j in range(1, num_steps + 1):
                z = np.random.standard_normal()
                price_path[j] = price_path[j-1] * np.exp(drift + volatility * z)
            
            final_prices[i] = price_path[-1]
        
        # Calculate payoffs at expiration
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put option
            payoffs = np.maximum(K - final_prices, 0)
        
        # Calculate present value of average payoff
        option_price = np.exp(-self.risk_free_rate * T) * np.mean(payoffs)
        
        return option_price
    
    def binomial_tree(self, S, K, T, sigma, option_type, is_american, num_steps=100):
        """
        Binomial tree model for option pricing
        
        Args:
            S (float): Current price of the underlying asset
            K (float): Strike price
            T (float): Time to expiration in years
            sigma (float): Implied volatility
            option_type (str): 'call' or 'put'
            is_american (bool): True for American options, False for European
            num_steps (int): Number of time steps in the tree
            
        Returns:
            float: Option price
        """
        if sigma <= 0 or T <= 0:
            return 0.0
        
        # Time step
        dt = T / num_steps
        
        # Up and down factors
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (np.exp((self.risk_free_rate - self.dividend_yield) * dt) - d) / (u - d)
        
        # Discount factor
        df = np.exp(-self.risk_free_rate * dt)
        
        # Initialize asset prices at maturity (final column of the tree)
        prices = np.zeros(num_steps + 1)
        for i in range(num_steps + 1):
            prices[i] = S * (u ** (num_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(num_steps + 1)
        if option_type == 'call':
            for i in range(num_steps + 1):
                option_values[i] = max(0, prices[i] - K)
        else:  # put option
            for i in range(num_steps + 1):
                option_values[i] = max(0, K - prices[i])
        
        # Work backwards through the tree
        for step in range(num_steps - 1, -1, -1):
            for i in range(step + 1):
                # Price of the underlying at this node
                price = S * (u ** (step - i)) * (d ** i)
                
                # Option value from discounted expected value
                option_values[i] = df * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                # For American options, check if early exercise is optimal
                if is_american:
                    if option_type == 'call':
                        option_values[i] = max(option_values[i], price - K)
                    else:  # put option
                        option_values[i] = max(option_values[i], K - price)
        
        return option_values[0]

    def calculate_greeks(self, option_data):
        """
        Calculate option Greeks using finite differences
        
        Args:
            option_data (dict): Option data containing required parameters
            
        Returns:
            dict: Dictionary containing delta, gamma, theta, vega, and rho
        """
        # Extract option parameters
        S = option_data['underlying_price']
        K = option_data['strike']
        T = option_data['time_to_expiry']
        sigma = option_data['implied_volatility']
        option_type = option_data['option_type'].lower()
        
        # Small change amounts for finite differences
        dS = S * 0.01  # 1% of stock price
        dsigma = 0.01  # 1% volatility
        dT = 1/365.0  # 1 day
        dr = 0.0001  # 1 basis point
        
        # Base price
        base_price = self.black_scholes(S, K, T, sigma, option_type)
        
        # Delta: dPrice/dS
        delta = (self.black_scholes(S + dS, K, T, sigma, option_type) - 
                 self.black_scholes(S - dS, K, T, sigma, option_type)) / (2 * dS)
        
        # Gamma: d²Price/dS²
        gamma = (self.black_scholes(S + dS, K, T, sigma, option_type) - 
                 2 * base_price + 
                 self.black_scholes(S - dS, K, T, sigma, option_type)) / (dS ** 2)
        
        # Theta: -dPrice/dT (negative, as T decreases with time)
        theta = -(self.black_scholes(S, K, T + dT, sigma, option_type) - 
                  base_price) / dT
        
        # Vega: dPrice/dσ (expressed per 1% change in volatility)
        vega = (self.black_scholes(S, K, T, sigma + dsigma, option_type) - 
                self.black_scholes(S, K, T, sigma - dsigma, option_type)) / (2 * dsigma) * 0.01
        
        # Rho: dPrice/dr (expressed per 1% change in interest rate)
        original_rate = self.risk_free_rate
        self.risk_free_rate = original_rate + dr
        price_up = self.black_scholes(S, K, T, sigma, option_type)
        self.risk_free_rate = original_rate - dr
        price_down = self.black_scholes(S, K, T, sigma, option_type)
        self.risk_free_rate = original_rate  # Restore original rate
        rho = (price_up - price_down) / (2 * dr) * 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
