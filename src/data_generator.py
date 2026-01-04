"""
Advanced Data Generator Module
Generates realistic synthetic datasets for ML analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except:
    pass


class DataGenerator:
    """Generate realistic synthetic datasets for ML demonstrations"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def generate_house_price_data(self, n_samples=2000):
        """
        Generate synthetic house price data for Regression
        
        Features:
        - square_feet: Living area (800-5000 sq ft)
        - bedrooms: Number of bedrooms (1-6)
        - bathrooms: Number of bathrooms (1-4)
        - age_years: Age of house (0-80 years)
        - distance_to_center_miles: Distance to city (1-50 miles)
        - has_pool: Binary (0/1)
        - has_garage: Binary (0/1)
        - neighborhood_score: Quality score (1-10)
        - lot_size_sqft: Lot size (2000-20000 sq ft)
        - stories: Number of floors (1-3)
        
        Target: price
        """
        print("\n    Generating House Price Dataset...")
        
        # Generate features with realistic distributions
        sqft = np.random.lognormal(mean=7.5, sigma=0.3, size=n_samples).clip(800, 6000)
        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
        bathrooms = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.45, 0.30, 0.10])
        age = np.random.exponential(scale=20, size=n_samples).clip(0, 80).astype(int)
        distance = np.random.exponential(scale=12, size=n_samples).clip(1, 50)
        has_pool = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        has_garage = np.random.choice([0, 1], n_samples, p=[0.30, 0.70])
        neighborhood = np.random.randint(1, 11, n_samples)
        lot_size = np.random.lognormal(mean=8.5, sigma=0.4, size=n_samples).clip(2000, 25000)
        stories = np.random.choice([1, 2, 3], n_samples, p=[0.40, 0.50, 0.10])
        
        # Generate price with complex relationships
        price = (
            30000 +  # Base price
            sqft * 120 +  # Price per sqft
            bedrooms * 12000 +  # Per bedroom
            bathrooms * 8000 +  # Per bathroom
            (80 - age) * 800 +  # Newer is better
            (35 - distance) * 1500 +  # Closer is better
            has_pool * 28000 +  # Pool premium
            has_garage * 15000 +  # Garage premium
            neighborhood * 18000 +  # Location quality
            lot_size * 3 +  # Lot size value
            stories * 10000 +  # Multi-story premium
            sqft * bedrooms * 5 +  # Interaction term
            np.random.normal(0, 25000, n_samples)  # Random noise
        ).clip(50000, 2500000)
        
        df = pd.DataFrame({
            'square_feet': sqft.astype(int),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age_years': age,
            'distance_to_center_miles': np.round(distance, 2),
            'has_pool': has_pool,
            'has_garage': has_garage,
            'neighborhood_score': neighborhood,
            'lot_size_sqft': lot_size.astype(int),
            'stories': stories,
            'price': np.round(price, 2)
        })
        
        filepath = os.path.join(self.data_dir, 'house_prices.csv')
        df.to_csv(filepath, index=False)
        print(f"    [+] House price data: {n_samples} samples, {len(df.columns)} features")
        print(f"        Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        print(f"        Saved to: {filepath}")
        
        return df
    
    def generate_customer_churn_data(self, n_samples=3000):
        """
        Generate synthetic customer churn data for Classification
        
        Features:
        - tenure_months: Months as customer (1-72)
        - monthly_charges: Monthly bill (20-150)
        - total_charges: Total paid
        - contract_type: Month-to-month, 1yr, 2yr
        - payment_method: Electronic, Mailed, Bank, Credit
        - tech_support: Yes/No
        - online_security: Yes/No
        - online_backup: Yes/No
        - device_protection: Yes/No
        - num_complaints: Complaints filed (0-10)
        - support_calls: Support calls made (0-20)
        
        Target: churn (0/1)
        """
        print("\n    Generating Customer Churn Dataset...")
        
        # Generate features
        tenure = np.random.randint(1, 73, n_samples)
        monthly_charges = np.random.uniform(20, 150, n_samples)
        total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
        total_charges = total_charges.clip(0)
        
        contract_type = np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            n_samples, p=[0.55, 0.25, 0.20]
        )
        payment_method = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
            n_samples, p=[0.35, 0.20, 0.25, 0.20]
        )
        tech_support = np.random.choice(['Yes', 'No'], n_samples, p=[0.40, 0.60])
        online_security = np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65])
        online_backup = np.random.choice(['Yes', 'No'], n_samples, p=[0.38, 0.62])
        device_protection = np.random.choice(['Yes', 'No'], n_samples, p=[0.42, 0.58])
        num_complaints = np.random.poisson(1.2, n_samples).clip(0, 10)
        support_calls = np.random.poisson(3, n_samples).clip(0, 20)
        
        # Calculate churn probability
        churn_prob = (
            0.25 +  # Base probability
            (tenure < 12) * 0.20 +  # Short tenure risk
            (monthly_charges > 85) * 0.12 +  # High charges risk
            (contract_type == 'Month-to-month') * 0.18 +  # Monthly contract risk
            (payment_method == 'Electronic check') * 0.08 +  # Payment method risk
            (tech_support == 'No') * 0.10 +  # No support risk
            (online_security == 'No') * 0.08 +  # No security risk
            (num_complaints > 2) * 0.15 +  # Complaints risk
            (support_calls > 6) * 0.10 -  # Many calls can indicate issues
            (tenure > 48) * 0.25 -  # Long tenure loyalty
            (contract_type == 'Two year') * 0.20 -  # Long contract loyalty
            (online_backup == 'Yes') * 0.05 -  # Backup users less likely to leave
            (device_protection == 'Yes') * 0.05  # Protection users invested
        )
        churn_prob = churn_prob.clip(0.05, 0.90)
        churn = (np.random.random(n_samples) < churn_prob).astype(int)
        
        df = pd.DataFrame({
            'tenure_months': tenure,
            'monthly_charges': np.round(monthly_charges, 2),
            'total_charges': np.round(total_charges, 2),
            'contract_type': contract_type,
            'payment_method': payment_method,
            'tech_support': tech_support,
            'online_security': online_security,
            'online_backup': online_backup,
            'device_protection': device_protection,
            'num_complaints': num_complaints,
            'support_calls': support_calls,
            'churn': churn
        })
        
        churn_rate = churn.sum() / len(churn) * 100
        
        filepath = os.path.join(self.data_dir, 'customer_churn.csv')
        df.to_csv(filepath, index=False)
        print(f"    [+] Customer churn data: {n_samples} samples, {len(df.columns)-1} features")
        print(f"        Churn rate: {churn_rate:.1f}%")
        print(f"        Saved to: {filepath}")
        
        return df
    
    def generate_sales_time_series(self, n_days=1095):
        """
        Generate synthetic sales time series (3 years)
        
        Components:
        - Trend: Upward growth
        - Weekly seasonality: Weekend peaks
        - Yearly seasonality: Summer & holiday peaks
        - Special events: Black Friday, Christmas
        - Random noise
        """
        print("\n    Generating Sales Time Series...")
        
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        t = np.arange(n_days)
        
        # Trend: Gradual growth with slight acceleration
        trend = 1000 + t * 0.8 + (t ** 1.1) * 0.001
        
        # Weekly seasonality (weekends higher)
        weekly = 120 * np.sin(2 * np.pi * t / 7 - np.pi/2)
        
        # Yearly seasonality
        yearly = (
            180 * np.sin(2 * np.pi * t / 365.25 - np.pi/4) +  # Summer peak
            120 * np.cos(4 * np.pi * t / 365.25)  # Additional pattern
        )
        
        # Special events
        special_events = np.zeros(n_days)
        for i, date in enumerate(dates):
            # December (holiday shopping)
            if date.month == 12:
                if date.day >= 20:
                    special_events[i] = 500  # Christmas week
                else:
                    special_events[i] = 250  # December
            # Black Friday (late November)
            elif date.month == 11 and date.day >= 24 and date.day <= 30:
                special_events[i] = 400
            # Summer sales (July)
            elif date.month == 7:
                special_events[i] = 100
            # Back to school (late August)
            elif date.month == 8 and date.day >= 15:
                special_events[i] = 80
        
        # Random noise with varying volatility
        noise = np.random.normal(0, 80 + t * 0.02, n_days)
        
        # Combine components
        sales = trend + weekly + yearly + special_events + noise
        sales = sales.clip(200)  # Minimum sales
        
        df = pd.DataFrame({
            'date': dates,
            'sales': np.round(sales, 2),
            'day_of_week': [d.strftime('%A') for d in dates],
            'month': [d.month for d in dates],
            'year': [d.year for d in dates],
            'quarter': [(d.month - 1) // 3 + 1 for d in dates],
            'is_weekend': [1 if d.weekday() >= 5 else 0 for d in dates],
            'is_holiday_season': [1 if d.month in [11, 12] else 0 for d in dates]
        })
        
        filepath = os.path.join(self.data_dir, 'sales_timeseries.csv')
        df.to_csv(filepath, index=False)
        print(f"    [+] Sales time series: {n_days} days ({n_days/365:.1f} years)")
        print(f"        Period: {dates[0].date()} to {dates[-1].date()}")
        print(f"        Sales range: ${df['sales'].min():,.0f} - ${df['sales'].max():,.0f}")
        print(f"        Saved to: {filepath}")
        
        return df
    
    def generate_all_datasets(self):
        """Generate all datasets"""
        print("\n" + "="*60)
        print("  DATA GENERATION")
        print("="*60)
        
        house_df = self.generate_house_price_data(n_samples=2000)
        churn_df = self.generate_customer_churn_data(n_samples=3000)
        sales_df = self.generate_sales_time_series(n_days=1095)
        
        print("\n" + "-"*60)
        print("  [+] All datasets generated successfully!")
        print(f"      Data directory: {self.data_dir}")
        print("="*60)
        
        return {
            'house_prices': house_df,
            'customer_churn': churn_df,
            'sales_timeseries': sales_df
        }


if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_all_datasets()
