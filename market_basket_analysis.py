"""
Market Basket Analysis Logic Module

This module contains all the business logic for market basket analysis,
including data processing, algorithm execution, and calculations.
"""

import pandas as pd
import numpy as np
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


class MarketBasketAnalyzer:
    """Main class for performing market basket analysis"""
    
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with a CSV file
        
        Args:
            csv_file_path (str): Path to the CSV file containing transaction data
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.transactions = None
        self.basket = None
        self.frequent_itemsets = None
        self.rules = None
        
    def load_data(self):
        """Load and preprocess the transaction data"""
        self.df = pd.read_csv(self.csv_file_path)
        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
        return self.df
    
    def prepare_transactions(self):
        """Convert dataframe to transaction format (list of lists)"""
        if self.df is None:
            self.load_data()
            
        transactions = []
        for _, row in self.df.iterrows():
            transaction = []
            for field in ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']:
                if pd.notna(row[field]) and row[field] != '':
                    transaction.append(row[field])
            if transaction:
                transactions.append(transaction)
        
        self.transactions = transactions
        return transactions
    
    def create_basket_matrix(self):
        """Create one-hot encoded basket matrix from transactions"""
        if self.transactions is None:
            self.prepare_transactions()
            
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        self.basket = pd.DataFrame(te_ary, columns=te.columns_)
        return self.basket
    
    def get_frequent_itemsets(self, min_support=0.01, algorithm='apriori'):
        """
        Generate frequent itemsets using specified algorithm
        
        Args:
            min_support (float): Minimum support threshold (0-1)
            algorithm (str): 'apriori' or 'fpgrowth'
            
        Returns:
            pd.DataFrame: Frequent itemsets with support values
        """
        if self.basket is None:
            self.create_basket_matrix()
        
        if algorithm == 'apriori':
            frequent_itemsets = apriori(self.basket, min_support=min_support, use_colnames=True)
        else:  # fpgrowth
            frequent_itemsets = fpgrowth(self.basket, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        
        self.frequent_itemsets = frequent_itemsets
        return frequent_itemsets
    
    def generate_association_rules(self, metric='lift', min_threshold=1.0):
        """
        Generate association rules from frequent itemsets
        
        Args:
            metric (str): Metric for filtering ('confidence', 'lift', 'support')
            min_threshold (float): Minimum threshold for the metric
            
        Returns:
            pd.DataFrame: Association rules with metrics
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            return pd.DataFrame()
        
        rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        if len(rules) > 0:
            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        self.rules = rules
        return rules
    
    def get_dataset_stats(self):
        """
        Get basic statistics about the dataset
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        if self.df is None:
            self.load_data()
        if self.transactions is None:
            self.prepare_transactions()
            
        all_products = set()
        for t in self.transactions:
            all_products.update(t)
        
        avg_items = np.mean([len(t) for t in self.transactions])
        date_range = (self.df['created_at'].max() - self.df['created_at'].min()).days
        
        return {
            'total_transactions': len(self.transactions),
            'unique_products': len(all_products),
            'avg_items_per_transaction': avg_items,
            'date_range_days': date_range,
            'start_date': self.df['created_at'].min(),
            'end_date': self.df['created_at'].max()
        }
    
    def get_product_counts(self):
        """
        Get purchase frequency for all products
        
        Returns:
            pd.DataFrame: Products with their purchase counts
        """
        if self.df is None:
            self.load_data()
            
        product_counts = {}
        
        for _, row in self.df.iterrows():
            for field in ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']:
                if pd.notna(row[field]) and row[field] != '':
                    product = row[field]
                    product_counts[product] = product_counts.get(product, 0) + 1
        
        products_df = pd.DataFrame(list(product_counts.items()), columns=['Product', 'Count'])
        products_df = products_df.sort_values('Count', ascending=False)
        
        return products_df
    
    def get_top_products(self, top_n=20):
        """
        Get top N most frequently purchased products
        
        Args:
            top_n (int): Number of top products to return
            
        Returns:
            pd.DataFrame: Top N products with counts
        """
        products_df = self.get_product_counts()
        return products_df.head(top_n)
    
    def calculate_cooccurrence_matrix(self, top_n=30):
        """
        Calculate product co-occurrence matrix
        
        Args:
            top_n (int): Number of top products to include
            
        Returns:
            tuple: (cooccurrence_matrix, product_names)
        """
        if self.transactions is None:
            self.prepare_transactions()
        
        # Get top products
        products_df = self.get_product_counts()
        top_product_names = products_df.head(top_n)['Product'].tolist()
        
        # Create co-occurrence matrix
        cooccurrence = np.zeros((len(top_product_names), len(top_product_names)))
        
        for transaction in self.transactions:
            transaction_products = [p for p in transaction if p in top_product_names]
            for i, prod1 in enumerate(top_product_names):
                for j, prod2 in enumerate(top_product_names):
                    if prod1 in transaction_products and prod2 in transaction_products and i != j:
                        cooccurrence[i, j] += 1
        
        return cooccurrence, top_product_names
    
    def build_association_network(self, top_n=50):
        """
        Build a network graph from association rules
        
        Args:
            top_n (int): Number of top rules to include
            
        Returns:
            networkx.DiGraph: Network graph of associations
        """
        if self.rules is None or len(self.rules) == 0:
            return None
        
        # Select top rules by lift
        top_rules = self.rules.nlargest(top_n, 'lift')
        
        # Create network
        G = nx.DiGraph()
        
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            for ant in antecedents:
                for cons in consequents:
                    if G.has_edge(ant, cons):
                        G[ant][cons]['weight'] += rule['lift']
                    else:
                        G.add_edge(ant, cons, weight=rule['lift'], 
                                  confidence=rule['confidence'], 
                                  support=rule['support'])
        
        return G
    
    def get_time_based_data(self):
        """
        Extract time-based transaction data
        
        Returns:
            dict: Dictionary containing time-based dataframes
        """
        if self.df is None:
            self.load_data()
        
        df_copy = self.df.copy()
        df_copy['date'] = df_copy['created_at'].dt.date
        df_copy['hour'] = df_copy['created_at'].dt.hour
        df_copy['day_of_week'] = df_copy['created_at'].dt.day_name()
        
        # Daily transactions
        daily_transactions = df_copy.groupby('date').size().reset_index(name='count')
        daily_transactions['date'] = pd.to_datetime(daily_transactions['date'])
        
        # Hourly transactions
        hourly_transactions = df_copy.groupby('hour').size().reset_index(name='count')
        
        # Day of week transactions
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_transactions = df_copy.groupby('day_of_week').size().reset_index(name='count')
        dow_transactions['day_of_week'] = pd.Categorical(
            dow_transactions['day_of_week'], 
            categories=day_order, 
            ordered=True
        )
        dow_transactions = dow_transactions.sort_values('day_of_week')
        
        # Peak hour analysis
        hourly_counts = df_copy.groupby('hour').size()
        peak_hour = hourly_counts.idxmax()
        peak_count = hourly_counts.max()
        avg_hourly = hourly_counts.mean()
        
        return {
            'daily': daily_transactions,
            'hourly': hourly_transactions,
            'day_of_week': dow_transactions,
            'peak_hour': peak_hour,
            'peak_count': peak_count,
            'avg_hourly': avg_hourly
        }
    
    def get_product_statistics(self):
        """
        Get detailed statistics for each product
        
        Returns:
            pd.DataFrame: Detailed product statistics
        """
        if self.transactions is None:
            self.prepare_transactions()
        
        product_stats = {}
        total_transactions = len(self.transactions)
        
        for transaction in self.transactions:
            for product in transaction:
                if product not in product_stats:
                    product_stats[product] = {
                        'frequency': 0,
                        'co_purchased_with': set()
                    }
                product_stats[product]['frequency'] += 1
                product_stats[product]['co_purchased_with'].update(
                    [p for p in transaction if p != product]
                )
        
        stats_df = pd.DataFrame([
            {
                'Product': product,
                'Purchase Frequency': stats['frequency'],
                'Support (%)': (stats['frequency'] / total_transactions) * 100,
                'Co-purchased with (unique)': len(stats['co_purchased_with'])
            }
            for product, stats in product_stats.items()
        ])
        
        stats_df = stats_df.sort_values('Purchase Frequency', ascending=False)
        stats_df = stats_df.round(2)
        
        return stats_df


class DataValidator:
    """Utility class for validating data and parameters"""
    
    @staticmethod
    def validate_support(support):
        """Validate support threshold"""
        if not 0 < support <= 1:
            raise ValueError("Support must be between 0 and 1")
        return True
    
    @staticmethod
    def validate_algorithm(algorithm):
        """Validate algorithm choice"""
        valid_algorithms = ['apriori', 'fpgrowth']
        if algorithm.lower() not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        return True
    
    @staticmethod
    def validate_metric(metric):
        """Validate metric choice"""
        valid_metrics = ['confidence', 'lift', 'support', 'conviction']
        if metric.lower() not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")
        return True


class RuleInterpreter:
    """Utility class for interpreting association rules"""
    
    @staticmethod
    def interpret_lift(lift):
        """
        Interpret lift value
        
        Args:
            lift (float): Lift value
            
        Returns:
            str: Interpretation of lift
        """
        if lift > 1.5:
            return "Strong positive correlation"
        elif lift > 1.0:
            return "Weak positive correlation"
        elif lift == 1.0:
            return "No correlation"
        else:
            return "Negative correlation"
    
    @staticmethod
    def interpret_confidence(confidence):
        """
        Interpret confidence value
        
        Args:
            confidence (float): Confidence value
            
        Returns:
            str: Interpretation of confidence
        """
        if confidence >= 0.8:
            return "Very high confidence"
        elif confidence >= 0.6:
            return "High confidence"
        elif confidence >= 0.4:
            return "Moderate confidence"
        else:
            return "Low confidence"
    
    @staticmethod
    def get_business_recommendation(rule):
        """
        Generate business recommendation from a rule
        
        Args:
            rule (pd.Series): Association rule
            
        Returns:
            str: Business recommendation
        """
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        
        recommendations = []
        
        if rule['lift'] > 1.5 and rule['confidence'] > 0.6:
            recommendations.append(f"Strong bundling opportunity: Offer {antecedents} with {consequents}")
        
        if rule['confidence'] > 0.7:
            recommendations.append(f"Cross-sell recommendation: Suggest {consequents} to customers buying {antecedents}")
        
        if rule['lift'] > 2.0:
            recommendations.append(f"Product placement: Display {antecedents} and {consequents} near each other")
        
        return " | ".join(recommendations) if recommendations else "Monitor this association"
