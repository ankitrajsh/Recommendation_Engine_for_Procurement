"""
Collaborative Filtering Recommendation Engine

This approach uses user-item interaction data to find products that are frequently
purchased together by similar users. It builds a product-product similarity matrix
based on cosine similarity of user purchase patterns.
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from psycopg2.extras import execute_batch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database_utils import fetch_purchase_data, get_category_data, store_recommendations
from utils.evaluation import RecommendationEvaluator
from config.database import RECOMMENDATION_CONFIG


class CollaborativeRecommender:
    """
    Collaborative filtering recommendation system based on user-item interactions.
    """
    
    def __init__(self):
        self.similarity_matrix = None
        self.product_ids = None
        self.product_map = None
        self.evaluator = RecommendationEvaluator()
        self.config = RECOMMENDATION_CONFIG
        
    def get_category_level_4(self, category_json):
        """Extract level 4 category from JSON."""
        try:
            if isinstance(category_json, str):
                categories = json.loads(category_json)
            else:
                categories = category_json

            if 'cat_4' in categories:
                cat_4_list = categories['cat_4']
                if cat_4_list and len(cat_4_list) > 0:
                    return str(cat_4_list[0].get('id'))
        except:
            pass
        return None
    
    def create_similarity_matrix(self, df):
        """
        Create product similarity matrix from purchase data.
        
        Args:
            df: DataFrame with purchase data
            
        Returns:
            tuple: (similarity_matrix, product_ids)
        """
        print("Creating similarity matrix...")

        # Create product-user purchase matrix
        purchase_matrix = df.pivot_table(
            index='product_id',
            columns='buyer_user_id',
            values='total_quantity',
            fill_value=0
        )

        # Calculate similarity between products
        similarity = cosine_similarity(purchase_matrix)

        return similarity, purchase_matrix.index.tolist()
    
    def fit(self):
        """
        Train the collaborative filtering model by building the similarity matrix.
        """
        print("Training Collaborative Filtering Recommender...")
        
        # Get purchase data
        purchase_df, self.dates_df = fetch_purchase_data()
        print(f"Found {len(purchase_df)} purchase records")

        # Generate similarity matrix
        self.similarity_matrix, self.product_ids = self.create_similarity_matrix(purchase_df)
        print(f"Generated similarities for {len(self.product_ids)} products")
        
        # Create product mapping for quick lookups
        self.product_map = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        # Print evaluation statistics
        sparsity_stats = self.evaluator.calculate_matrix_sparsity(
            purchase_df.pivot_table(index='product_id', columns='buyer_user_id', 
                                   values='total_quantity', fill_value=0)
        )
        print(f"Purchase matrix sparsity: {sparsity_stats['sparsity_percentage']:.2f}%")
        
        similarity_stats = self.evaluator.calculate_similarity_stats(self.similarity_matrix)
        print(f"Similarity matrix - Non-zero similarities: {similarity_stats['non_zero_similarities']} / {similarity_stats['total_similarities']} ({similarity_stats['non_zero_percentage']:.2f}%)")
        
        return self
    
    def get_recommendations(self, product_id, top_n=None, min_score=None):
        """
        Get recommendations for a specific product.
        
        Args:
            product_id: Product ID to get recommendations for
            top_n: Number of recommendations to return
            min_score: Minimum similarity score threshold
            
        Returns:
            list: List of (product_id, similarity_score) tuples
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        top_n = top_n or self.config['top_n_recommendations']
        min_score = min_score or self.config['similarity_threshold']
        
        if product_id not in self.product_map:
            return []
        
        product_idx = self.product_map[product_id]
        similarities = self.similarity_matrix[product_idx]
        
        # Get top similar products (excluding self)
        similar_indices = np.argsort(-similarities)
        recommendations = []
        
        for idx in similar_indices[1:]:  # Skip first (self)
            if len(recommendations) >= top_n:
                break
            
            similarity_score = similarities[idx]
            if similarity_score >= min_score:
                recommended_product_id = self.product_ids[idx]
                recommendations.append((recommended_product_id, similarity_score))
        
        return recommendations
    
    def generate_all_recommendations(self, top_n=None, min_score=None):
        """
        Generate recommendations for all products.
        
        Args:
            top_n: Number of recommendations per product
            min_score: Minimum similarity score threshold
            
        Returns:
            dict: Dictionary with product_id as key and recommendations as value
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        top_n = top_n or self.config['top_n_recommendations']
        min_score = min_score or self.config['similarity_threshold']
        
        print(f"Generating recommendations for {len(self.product_ids)} products...")
        
        all_recommendations = {}
        
        for product_id in self.product_ids:
            recommendations = self.get_recommendations(product_id, top_n, min_score)
            all_recommendations[product_id] = recommendations
        
        return all_recommendations
    
    def save_to_database(self, recommendations=None):
        """
        Save recommendations to the database.
        
        Args:
            recommendations: Optional pre-generated recommendations dict
        """
        if recommendations is None:
            recommendations = self.generate_all_recommendations()
        
        print(f"Saving recommendations to database...")
        
        # Get category data for all recommended products
        all_recommended_products = set()
        for recs in recommendations.values():
            for rec_id, _ in recs:
                all_recommended_products.add(rec_id)
        
        category_data = get_category_data(list(all_recommended_products))
        
        # Prepare recommendations for database insertion
        db_recommendations = []
        
        for current_product, recs in recommendations.items():
            if current_product not in self.dates_df['product_id'].values:
                continue
                
            product_dates = self.dates_df[self.dates_df['product_id'] == current_product].iloc[0]
            
            for similar_product, score in recs:
                cat_4 = self.get_category_level_4(category_data.get(similar_product, '{}'))
                if cat_4:
                    db_recommendations.append((
                        int(current_product),          # id
                        cat_4,                         # category_id
                        int(similar_product),          # product_id
                        'collaborative_filtering',     # product_type
                        'A',                          # status
                        product_dates['created_date'],
                        product_dates['updated_date']
                    ))
        
        if db_recommendations:
            store_recommendations(db_recommendations)
            print(f"Saved {len(db_recommendations)} recommendations to database")
        else:
            print("No valid recommendations to save")
    
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
        
        return self.evaluator.generate_evaluation_report(
            "Collaborative Filtering",
            recommendations,
            self.similarity_matrix,
            self.product_map,
            len(self.product_ids)
        )


def main():
    """
    Main function to demonstrate the collaborative filtering recommender.
    """
    print("Starting Collaborative Filtering Recommendation System")
    print("=" * 60)
    
    # Initialize and train the recommender
    recommender = CollaborativeRecommender()
    recommender.fit()
    
    # Generate all recommendations
    recommendations = recommender.generate_all_recommendations()
    
    # Evaluate the system
    evaluation_report = recommender.evaluate(recommendations)
    recommender.evaluator.print_evaluation_report(evaluation_report)
    
    # Save to database
    recommender.save_to_database(recommendations)
    
    # Interactive testing
    print("\nInteractive Testing Mode")
    print("-" * 30)
    
    while True:
        try:
            choice = input("\nTest a specific product ID? (y/n): ").lower()
            if choice != 'y':
                break
            
            product_id = int(input("Enter product ID: "))
            recs = recommender.get_recommendations(product_id)
            
            if recs:
                print(f"\nTop recommendations for Product {product_id}:")
                for i, (rec_id, score) in enumerate(recs, 1):
                    print(f"  {i}. Product {rec_id} (similarity: {score:.4f})")
            else:
                print(f"No recommendations found for Product {product_id}")
                
        except ValueError:
            print("Invalid input. Please enter a valid product ID.")
        except KeyboardInterrupt:
            break
    
    print("\nCollaborative Filtering Recommender completed!")


if __name__ == "__main__":
    main()
