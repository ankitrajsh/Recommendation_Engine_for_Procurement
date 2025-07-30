"""
Content-Based Filtering Recommendation Engine

This approach uses co-occurrence analysis based on vendor catalogs to find products
that are commonly sold together by the same vendors. It analyzes vendor product
catalogs and recommends products based on vendor overlap patterns.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database_utils import fetch_vendor_product_data
from utils.evaluation import RecommendationEvaluator
from config.database import RECOMMENDATION_CONFIG


class ContentBasedRecommender:
    """
    Content-based filtering recommendation system using vendor co-occurrence analysis.
    """
    
    def __init__(self):
        self.product_df = None
        self.co_occurrence_dict = None
        self.product_to_vendors = {}
        self.vendor_to_products = {}
        self.evaluator = RecommendationEvaluator()
        self.config = RECOMMENDATION_CONFIG
        self.threshold = self.config.get('co_occurrence_threshold', 0.8)
        
    def build_lookup_dictionaries(self):
        """Build efficient lookup dictionaries for products and vendors."""
        print("Building lookup dictionaries...")
        
        for _, row in self.product_df.iterrows():
            vendor, product = row['vendor_name'], row['product_name']

            # Product to vendors mapping
            if product not in self.product_to_vendors:
                self.product_to_vendors[product] = set()
            self.product_to_vendors[product].add(vendor)

            # Vendor to products mapping
            if vendor not in self.vendor_to_products:
                self.vendor_to_products[vendor] = set()
            self.vendor_to_products[vendor].add(product)
    
    def calculate_co_occurrences(self):
        """Calculate co-occurrences using optimized set operations."""
        print("Calculating product co-occurrences...")
        self.co_occurrence_dict = defaultdict(dict)

        # Get products with enough vendors
        valid_products = [
            product for product, vendors in self.product_to_vendors.items()
            if len(vendors) >= 2
        ]

        print(f"Processing {len(valid_products)} products with multiple vendors...")

        # Calculate co-occurrences using set operations
        for product1 in tqdm(valid_products, desc="Calculating co-occurrences"):
            vendors_with_product1 = self.product_to_vendors[product1]
            vendors_count1 = len(vendors_with_product1)

            for product2 in valid_products:
                if product1 != product2:
                    vendors_with_product2 = self.product_to_vendors[product2]
                    # Fast set intersection
                    vendors_with_both = len(vendors_with_product1 & vendors_with_product2)

                    if vendors_with_both > 0:
                        percentage = vendors_with_both / vendors_count1
                        if percentage >= self.threshold:
                            self.co_occurrence_dict[product1][product2] = percentage
    
    def fit(self):
        """
        Train the content-based filtering model by building co-occurrence matrix.
        """
        print("Training Content-Based Filtering Recommender...")
        
        # Fetch vendor-product data
        self.product_df = fetch_vendor_product_data()
        print(f"Fetched {len(self.product_df)} unique vendor-product pairs")
        
        # Build lookup dictionaries
        self.build_lookup_dictionaries()
        
        # Calculate co-occurrences
        self.calculate_co_occurrences()
        
        # Print statistics
        total_products = len(self.product_to_vendors)
        products_with_recs = len(self.co_occurrence_dict)
        print(f"Generated co-occurrences for {products_with_recs} out of {total_products} products")
        
        return self
    
    def get_recommendations(self, vendor_name=None, product_name=None, top_n=None, min_confidence=None):
        """
        Get recommendations for a specific vendor-product combination.
        
        Args:
            vendor_name: Name of the vendor
            product_name: Name of the product
            top_n: Number of recommendations to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            list: List of recommendation dictionaries
        """
        if self.co_occurrence_dict is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        top_n = top_n or self.config['top_n_recommendations']
        min_confidence = min_confidence or self.threshold
        
        try:
            # Get current vendor's products using fast lookup
            current_products = self.vendor_to_products.get(vendor_name, set())

            recommendations = []
            if product_name in self.co_occurrence_dict:
                for related_product, percentage in self.co_occurrence_dict[product_name].items():
                    if related_product not in current_products and percentage >= min_confidence:
                        recommendations.append({
                            'product': related_product,
                            'confidence': percentage,
                            'message': f"{percentage*100:.1f}% of vendors who have '{product_name}' "
                                     f"also have '{related_product}'"
                        })

            # Sort by confidence and return top N
            recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
            return recommendations[:top_n]

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def get_recommendations_by_product(self, product_name, top_n=None, min_confidence=None):
        """
        Get recommendations for a product regardless of vendor.
        
        Args:
            product_name: Name of the product
            top_n: Number of recommendations to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            list: List of (product_name, confidence_score) tuples
        """
        if self.co_occurrence_dict is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        top_n = top_n or self.config['top_n_recommendations']
        min_confidence = min_confidence or self.threshold
        
        recommendations = []
        
        if product_name in self.co_occurrence_dict:
            for related_product, confidence in self.co_occurrence_dict[product_name].items():
                if confidence >= min_confidence:
                    recommendations.append((related_product, confidence))
        
        # Sort by confidence and return top N
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        return recommendations[:top_n]
    
    def generate_all_recommendations(self, top_n=None, min_confidence=None):
        """
        Generate recommendations for all products.
        
        Args:
            top_n: Number of recommendations per product
            min_confidence: Minimum confidence threshold
            
        Returns:
            dict: Dictionary with product_name as key and recommendations as value
        """
        if self.co_occurrence_dict is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        top_n = top_n or self.config['top_n_recommendations']
        min_confidence = min_confidence or self.threshold
        
        print(f"Generating recommendations for {len(self.co_occurrence_dict)} products...")
        
        all_recommendations = {}
        
        for product_name in self.co_occurrence_dict.keys():
            recommendations = self.get_recommendations_by_product(
                product_name, top_n, min_confidence
            )
            all_recommendations[product_name] = recommendations
        
        return all_recommendations
    
    def get_product_statistics(self):
        """
        Get statistics about the product catalog.
        
        Returns:
            dict: Statistics dictionary
        """
        if not self.product_to_vendors:
            return {}
        
        vendor_counts = [len(vendors) for vendors in self.product_to_vendors.values()]
        product_counts = [len(products) for products in self.vendor_to_products.values()]
        
        return {
            'total_products': len(self.product_to_vendors),
            'total_vendors': len(self.vendor_to_products),
            'avg_vendors_per_product': np.mean(vendor_counts),
            'avg_products_per_vendor': np.mean(product_counts),
            'max_vendors_per_product': max(vendor_counts),
            'max_products_per_vendor': max(product_counts),
            'products_with_multiple_vendors': sum(1 for count in vendor_counts if count > 1),
            'products_with_recommendations': len(self.co_occurrence_dict)
        }
    
    def evaluate(self, recommendations=None):
        """
        Evaluate the recommendation system performance.
        
        Args:
            recommendations: Optional pre-generated recommendations
            
        Returns:
            dict: Evaluation report
        """
        if recommendations is None:
            recommendations = self.generate_all_recommendations()
        
        # Convert to format expected by evaluator (product_id -> [(rec_id, score)])
        formatted_recs = {}
        for product, recs in recommendations.items():
            formatted_recs[product] = recs
        
        stats = self.get_product_statistics()
        
        report = self.evaluator.generate_evaluation_report(
            "Content-Based Filtering",
            formatted_recs,
            total_products=stats.get('total_products', len(self.product_to_vendors))
        )
        
        # Add content-specific statistics
        report['content_stats'] = stats
        
        return report
    
    def print_statistics(self):
        """Print detailed statistics about the content-based system."""
        stats = self.get_product_statistics()
        
        print(f"\n{'='*50}")
        print("CONTENT-BASED SYSTEM STATISTICS")
        print(f"{'='*50}")
        print(f"Total Products: {stats['total_products']}")
        print(f"Total Vendors: {stats['total_vendors']}")
        print(f"Average Vendors per Product: {stats['avg_vendors_per_product']:.2f}")
        print(f"Average Products per Vendor: {stats['avg_products_per_vendor']:.2f}")
        print(f"Products with Multiple Vendors: {stats['products_with_multiple_vendors']}")
        print(f"Products with Recommendations: {stats['products_with_recommendations']}")
        print(f"Co-occurrence Threshold: {self.threshold}")
        print(f"{'='*50}\n")


def main():
    """
    Main function to demonstrate the content-based filtering recommender.
    """
    print("Starting Content-Based Filtering Recommendation System")
    print("=" * 60)
    
    # Initialize and train the recommender
    recommender = ContentBasedRecommender()
    recommender.fit()
    
    # Print statistics
    recommender.print_statistics()
    
    # Generate all recommendations
    recommendations = recommender.generate_all_recommendations()
    
    # Evaluate the system
    evaluation_report = recommender.evaluate(recommendations)
    recommender.evaluator.print_evaluation_report(evaluation_report)
    
    # Interactive testing
    print("\nInteractive Testing Mode")
    print("-" * 30)
    
    while True:
        try:
            choice = input("\nTest recommendations? (y/n): ").lower()
            if choice != 'y':
                break
            
            vendor_name = input("Enter vendor name: ").strip()
            product_name = input("Enter product name: ").strip()
            
            recs = recommender.get_recommendations(vendor_name, product_name)
            
            if recs:
                print(f"\nRecommendations for '{product_name}' by '{vendor_name}':")
                for i, rec in enumerate(recs, 1):
                    print(f"  {i}. {rec['message']}")
            else:
                print(f"No recommendations found for '{product_name}' by '{vendor_name}'")
                
        except KeyboardInterrupt:
            break
    
    print("\nContent-Based Filtering Recommender completed!")


if __name__ == "__main__":
    main()
