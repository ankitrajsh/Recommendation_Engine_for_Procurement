"""
Hybrid Recommendation Engine

This approach combines collaborative filtering and content-based methods using matrix
factorization (NMF) and TF-IDF content similarity. It provides the best of both worlds
by leveraging user interaction data and product content features.
"""

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database_utils import fetch_detailed_interaction_data, store_recommendations
from utils.evaluation import RecommendationEvaluator
from config.database import RECOMMENDATION_CONFIG


class HybridRecommender:
    """
    Hybrid recommendation system combining collaborative and content-based filtering.
    """
    
    def __init__(self):
        self.interactions_df = None
        self.product_df = None
        self.vendor_to_master_map = None
        self.similarity_matrix = None
        self.product_map = None
        self.product_user_matrix = None
        self.evaluator = RecommendationEvaluator()
        self.config = RECOMMENDATION_CONFIG
        
        # Hybrid-specific parameters
        self.nmf_components = self.config.get('matrix_factorization_components', 50)
        self.collaborative_weight = 0.7
        self.content_weight = 0.3
        
    def create_product_interaction_matrix(self):
        """Create product-user interaction matrix using vendor_product_id."""
        print("Creating product-user interaction matrix...")
        
        self.product_user_matrix = self.interactions_df.pivot_table(
            index='vendor_product_id', 
            columns='user_id', 
            values='interaction_strength', 
            fill_value=0
        )
        
        return self.product_user_matrix
    
    def calculate_collaborative_similarity(self):
        """Calculate collaborative filtering similarity using cosine similarity."""
        print("Calculating collaborative filtering similarity...")
        
        # Use cosine similarity on the product-user matrix
        collaborative_sim = cosine_similarity(self.product_user_matrix)
        
        return collaborative_sim
    
    def calculate_content_similarity(self):
        """Calculate content-based similarity using product features."""
        print("Calculating content-based similarity...")
        
        # For this implementation, we'll use a simple approach since we don't have
        # rich product content. We'll create similarity based on master product grouping
        
        # Get unique vendor products and their master products
        vendor_products = self.product_user_matrix.index.tolist()
        
        # Create content similarity matrix
        n_products = len(vendor_products)
        content_sim = np.zeros((n_products, n_products))
        
        for i, prod1 in enumerate(vendor_products):
            for j, prod2 in enumerate(vendor_products):
                if i == j:
                    content_sim[i][j] = 1.0
                else:
                    # Products are similar if they belong to the same master product
                    master1 = self.vendor_to_master_map.get(prod1)
                    master2 = self.vendor_to_master_map.get(prod2)
                    
                    if master1 and master2 and master1 == master2:
                        content_sim[i][j] = 0.8  # High similarity for same master product
                    else:
                        content_sim[i][j] = 0.0  # No similarity otherwise
        
        return content_sim
    
    def calculate_nmf_similarity(self):
        """Calculate similarity using Non-negative Matrix Factorization."""
        print("Calculating NMF-based similarity...")
        
        # Apply NMF to the product-user matrix
        nmf = NMF(n_components=min(self.nmf_components, min(self.product_user_matrix.shape)), 
                  random_state=42, max_iter=200)
        
        try:
            # Fit NMF and get the product feature matrix
            W = nmf.fit_transform(self.product_user_matrix)
            
            # Calculate similarity based on NMF features
            nmf_similarity = cosine_similarity(W)
            
            print(f"NMF decomposition completed with {W.shape[1]} components")
            return nmf_similarity
            
        except Exception as e:
            print(f"NMF failed: {e}. Falling back to collaborative similarity.")
            return self.calculate_collaborative_similarity()
    
    def combine_similarities(self, collaborative_sim, content_sim):
        """Combine collaborative and content-based similarities."""
        print("Combining similarity matrices...")
        
        # Weighted combination of similarities
        combined_sim = (self.collaborative_weight * collaborative_sim + 
                       self.content_weight * content_sim)
        
        return combined_sim
    
    def fit(self):
        """
        Train the hybrid recommendation model.
        """
        print("Training Hybrid Recommendation System...")
        
        # Fetch detailed interaction data
        self.interactions_df, self.product_df, self.vendor_to_master_map = fetch_detailed_interaction_data()
        print(f"Loaded {len(self.interactions_df)} interactions for {len(self.product_df)} products")
        
        # Create product-user interaction matrix
        self.product_user_matrix = self.create_product_interaction_matrix()
        
        # Print matrix statistics
        sparsity_stats = self.evaluator.calculate_matrix_sparsity(self.product_user_matrix)
        print(f"Interaction matrix shape: {self.product_user_matrix.shape}")
        print(f"Matrix sparsity: {sparsity_stats['sparsity_percentage']:.2f}%")
        
        # Filter products with minimum interactions
        min_interactions = self.config.get('min_interactions', 1)
        row_sums = self.product_user_matrix.sum(axis=1)
        valid_products = row_sums[row_sums >= min_interactions].index
        print(f"Products with at least {min_interactions} interaction(s): {len(valid_products)} out of {len(self.product_user_matrix)}")
        
        # Keep only valid products
        self.product_user_matrix = self.product_user_matrix.loc[valid_products]
        
        # Create product mapping
        self.product_map = {product_id: idx for idx, product_id in enumerate(self.product_user_matrix.index)}
        
        # Calculate different similarity matrices
        collaborative_sim = self.calculate_collaborative_similarity()
        content_sim = self.calculate_content_similarity()
        
        # Use NMF for enhanced collaborative filtering
        nmf_sim = self.calculate_nmf_similarity()
        
        # Combine similarities (using NMF instead of basic collaborative)
        self.similarity_matrix = self.combine_similarities(nmf_sim, content_sim)
        
        # Print similarity statistics
        similarity_stats = self.evaluator.calculate_similarity_stats(self.similarity_matrix)
        print(f"Combined similarity matrix - Non-zero similarities: {similarity_stats['non_zero_similarities']} / {similarity_stats['total_similarities']} ({similarity_stats['non_zero_percentage']:.2f}%)")
        
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
                recommended_product_id = list(self.product_map.keys())[idx]
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
        
        print(f"Generating hybrid recommendations for {len(self.product_map)} products...")
        
        all_recommendations = {}
        
        for product_id in self.product_map.keys():
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
        
        print(f"Saving hybrid recommendations to database...")
        
        # Prepare recommendations for database insertion
        db_recommendations = []
        
        for current_product, recs in recommendations.items():
            # Get master product for current product
            master_product = self.vendor_to_master_map.get(current_product)
            if not master_product:
                continue
            
            for similar_product, score in recs:
                # Get master product for recommended product
                rec_master_product = self.vendor_to_master_map.get(similar_product)
                if rec_master_product:
                    db_recommendations.append((
                        int(master_product),           # id (using master product as ID)
                        str(rec_master_product),       # category_id (using recommended master product)
                        int(similar_product),          # product_id (vendor product)
                        'hybrid_filtering',            # product_type
                        'A',                          # status
                        datetime.now(),               # created_date
                        datetime.now()                # updated_date
                    ))
        
        if db_recommendations:
            store_recommendations(db_recommendations)
            print(f"Saved {len(db_recommendations)} hybrid recommendations to database")
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
            "Hybrid Filtering",
            recommendations,
            self.similarity_matrix,
            self.product_map,
            len(self.product_map)
        )
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'product_map': self.product_map,
            'vendor_to_master_map': self.vendor_to_master_map,
            'config': self.config,
            'collaborative_weight': self.collaborative_weight,
            'content_weight': self.content_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.similarity_matrix = model_data['similarity_matrix']
        self.product_map = model_data['product_map']
        self.vendor_to_master_map = model_data['vendor_to_master_map']
        self.config = model_data.get('config', self.config)
        self.collaborative_weight = model_data.get('collaborative_weight', 0.7)
        self.content_weight = model_data.get('content_weight', 0.3)
        
        print(f"Model loaded from {filepath}")
        return self


def main():
    """
    Main function to demonstrate the hybrid recommender.
    """
    print("Starting Hybrid Recommendation System")
    print("=" * 60)
    
    # Initialize and train the recommender
    recommender = HybridRecommender()
    recommender.fit()
    
    # Generate all recommendations
    recommendations = recommender.generate_all_recommendations()
    
    # Evaluate the system
    evaluation_report = recommender.evaluate(recommendations)
    recommender.evaluator.print_evaluation_report(evaluation_report)
    
    # Save to database
    recommender.save_to_database(recommendations)
    
    # Save model for future use
    model_path = "hybrid_model.pkl"
    recommender.save_model(model_path)
    
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
                print(f"\nTop hybrid recommendations for Product {product_id}:")
                for i, (rec_id, score) in enumerate(recs, 1):
                    master_rec = recommender.vendor_to_master_map.get(rec_id, 'Unknown')
                    print(f"  {i}. Product {rec_id} (Master: {master_rec}, Score: {score:.4f})")
            else:
                print(f"No recommendations found for Product {product_id}")
                
        except ValueError:
            print("Invalid input. Please enter a valid product ID.")
        except KeyboardInterrupt:
            break
    
    print("\nHybrid Recommender completed!")


if __name__ == "__main__":
    main()
