"""
Sample data generation utilities for testing recommendation engines.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


class SampleDataGenerator:
    """Generate sample data for testing recommendation engines."""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_sample_products(self, n_products=100):
        """
        Generate sample product data.
        
        Args:
            n_products: Number of products to generate
            
        Returns:
            pd.DataFrame: Sample product data
        """
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Toys']
        brands = ['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D', 'Brand_E']
        
        products = []
        for i in range(n_products):
            product = {
                'product_id': i + 1,
                'product_name': f'Product_{i+1}',
                'category': random.choice(categories),
                'brand': random.choice(brands),
                'price': round(random.uniform(10, 1000), 2),
                'created_date': datetime.now() - timedelta(days=random.randint(1, 365))
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_sample_users(self, n_users=50):
        """
        Generate sample user data.
        
        Args:
            n_users: Number of users to generate
            
        Returns:
            pd.DataFrame: Sample user data
        """
        users = []
        for i in range(n_users):
            user = {
                'user_id': i + 1,
                'user_name': f'User_{i+1}',
                'age_group': random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                'location': random.choice(['City_A', 'City_B', 'City_C', 'City_D']),
                'join_date': datetime.now() - timedelta(days=random.randint(30, 730))
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_sample_interactions(self, n_products=100, n_users=50, n_interactions=1000):
        """
        Generate sample user-product interactions.
        
        Args:
            n_products: Number of products
            n_users: Number of users
            n_interactions: Number of interactions to generate
            
        Returns:
            pd.DataFrame: Sample interaction data
        """
        interactions = []
        
        for _ in range(n_interactions):
            interaction = {
                'user_id': random.randint(1, n_users),
                'product_id': random.randint(1, n_products),
                'interaction_type': random.choice(['view', 'purchase', 'cart_add']),
                'quantity': random.randint(1, 5),
                'rating': random.choice([None, 1, 2, 3, 4, 5]),
                'timestamp': datetime.now() - timedelta(days=random.randint(1, 90))
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def generate_sample_vendor_catalog(self, n_vendors=20, n_products=100):
        """
        Generate sample vendor-product catalog data.
        
        Args:
            n_vendors: Number of vendors
            n_products: Number of products
            
        Returns:
            pd.DataFrame: Sample vendor catalog data
        """
        vendors = [f'Vendor_{i+1}' for i in range(n_vendors)]
        product_names = [f'Product_{i+1}' for i in range(n_products)]
        
        catalog = []
        
        # Each vendor carries a random subset of products
        for vendor in vendors:
            n_vendor_products = random.randint(10, 50)  # Each vendor has 10-50 products
            vendor_products = random.sample(product_names, n_vendor_products)
            
            for product in vendor_products:
                catalog_entry = {
                    'vendor_name': vendor,
                    'product_name': product,
                    'vendor_price': round(random.uniform(10, 1000), 2),
                    'availability': random.choice(['In Stock', 'Out of Stock', 'Limited']),
                    'last_updated': datetime.now() - timedelta(days=random.randint(1, 30))
                }
                catalog.append(catalog_entry)
        
        return pd.DataFrame(catalog)
    
    def generate_complete_sample_dataset(self, n_products=100, n_users=50, 
                                       n_interactions=1000, n_vendors=20):
        """
        Generate a complete sample dataset for testing.
        
        Args:
            n_products: Number of products
            n_users: Number of users
            n_interactions: Number of interactions
            n_vendors: Number of vendors
            
        Returns:
            dict: Dictionary containing all sample datasets
        """
        print("Generating complete sample dataset...")
        
        datasets = {
            'products': self.generate_sample_products(n_products),
            'users': self.generate_sample_users(n_users),
            'interactions': self.generate_sample_interactions(n_products, n_users, n_interactions),
            'vendor_catalog': self.generate_sample_vendor_catalog(n_vendors, n_products)
        }
        
        print(f"Generated:")
        print(f"  - {len(datasets['products'])} products")
        print(f"  - {len(datasets['users'])} users")
        print(f"  - {len(datasets['interactions'])} interactions")
        print(f"  - {len(datasets['vendor_catalog'])} vendor catalog entries")
        
        return datasets
    
    def save_sample_data(self, datasets, output_dir='sample_data'):
        """
        Save sample datasets to CSV files.
        
        Args:
            datasets: Dictionary of datasets
            output_dir: Output directory for CSV files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in datasets.items():
            filepath = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filepath, index=False)
            print(f"Saved {name} data to {filepath}")
    
    def create_similarity_test_data(self):
        """
        Create specific test data to verify similarity calculations.
        
        Returns:
            dict: Test datasets with known similarity patterns
        """
        # Create products with known relationships
        products = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
            'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics']
        })
        
        # Create users with specific purchase patterns
        interactions = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5],
            'product_id': [1, 2, 3, 1, 2, 4, 1, 3, 2, 4, 3, 5],
            'quantity': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'interaction_type': ['purchase'] * 12
        })
        
        # Create vendor catalog with known co-occurrences
        vendor_catalog = pd.DataFrame({
            'vendor_name': ['TechVendor', 'TechVendor', 'TechVendor', 'TechVendor',
                           'ElectroShop', 'ElectroShop', 'ElectroShop',
                           'GadgetStore', 'GadgetStore', 'GadgetStore'],
            'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor',
                           'Laptop', 'Mouse', 'Headphones',
                           'Keyboard', 'Monitor', 'Headphones']
        })
        
        return {
            'products': products,
            'interactions': interactions,
            'vendor_catalog': vendor_catalog
        }


def main():
    """Demo function to generate and save sample data."""
    print("Sample Data Generator Demo")
    print("=" * 40)
    
    generator = SampleDataGenerator()
    
    # Generate complete sample dataset
    datasets = generator.generate_complete_sample_dataset(
        n_products=50,
        n_users=25,
        n_interactions=500,
        n_vendors=10
    )
    
    # Save to CSV files
    generator.save_sample_data(datasets)
    
    # Generate test data for similarity verification
    test_data = generator.create_similarity_test_data()
    print("\nGenerated similarity test data:")
    print(f"  - {len(test_data['products'])} test products")
    print(f"  - {len(test_data['interactions'])} test interactions")
    print(f"  - {len(test_data['vendor_catalog'])} test vendor entries")
    
    generator.save_sample_data(test_data, 'test_data')
    
    print("\nSample data generation completed!")


if __name__ == "__main__":
    main()
