"""
Evaluation utilities for recommendation engines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time


class RecommendationEvaluator:
    """Utility class for evaluating recommendation systems."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_matrix_sparsity(self, matrix):
        """
        Calculate sparsity of a matrix.
        
        Args:
            matrix: Input matrix (numpy array or pandas DataFrame)
        
        Returns:
            dict: Sparsity statistics
        """
        if hasattr(matrix, 'values'):
            matrix = matrix.values
        
        total_cells = matrix.shape[0] * matrix.shape[1]
        non_zero_cells = np.count_nonzero(matrix)
        sparsity = 100 * (1 - non_zero_cells / total_cells)
        
        return {
            'total_cells': total_cells,
            'non_zero_cells': non_zero_cells,
            'sparsity_percentage': sparsity,
            'density_percentage': 100 - sparsity
        }
    
    def calculate_similarity_stats(self, similarity_matrix):
        """
        Calculate statistics for a similarity matrix.
        
        Args:
            similarity_matrix: Square similarity matrix
        
        Returns:
            dict: Similarity statistics
        """
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        off_diagonal = similarity_matrix[mask]
        
        non_zero_similarities = np.count_nonzero(off_diagonal)
        total_similarities = len(off_diagonal)
        
        return {
            'total_similarities': total_similarities,
            'non_zero_similarities': non_zero_similarities,
            'non_zero_percentage': (non_zero_similarities / total_similarities) * 100,
            'mean_similarity': np.mean(off_diagonal),
            'max_similarity': np.max(off_diagonal),
            'min_similarity': np.min(off_diagonal),
            'std_similarity': np.std(off_diagonal)
        }
    
    def evaluate_coverage(self, recommendations, total_products):
        """
        Evaluate recommendation coverage.
        
        Args:
            recommendations: Dict of product_id -> list of recommendations
            total_products: Total number of products in catalog
        
        Returns:
            dict: Coverage metrics
        """
        products_with_recs = len([p for p, recs in recommendations.items() if recs])
        recommended_products = set()
        
        for recs in recommendations.values():
            for rec_id, _ in recs:
                recommended_products.add(rec_id)
        
        return {
            'catalog_coverage': len(recommended_products) / total_products * 100,
            'prediction_coverage': products_with_recs / len(recommendations) * 100,
            'unique_recommended_products': len(recommended_products),
            'products_with_recommendations': products_with_recs
        }
    
    def calculate_diversity(self, recommendations, similarity_matrix=None, product_map=None):
        """
        Calculate diversity of recommendations.
        
        Args:
            recommendations: Dict of product_id -> list of recommendations
            similarity_matrix: Optional similarity matrix for diversity calculation
            product_map: Optional mapping from product_id to matrix index
        
        Returns:
            dict: Diversity metrics
        """
        if similarity_matrix is None or product_map is None:
            return {'diversity_score': None, 'note': 'Similarity matrix required for diversity calculation'}
        
        diversity_scores = []
        
        for product_id, recs in recommendations.items():
            if len(recs) < 2:
                continue
            
            # Calculate pairwise similarities within recommendations
            rec_similarities = []
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    rec_id1, _ = recs[i]
                    rec_id2, _ = recs[j]
                    
                    if rec_id1 in product_map and rec_id2 in product_map:
                        idx1 = product_map[rec_id1]
                        idx2 = product_map[rec_id2]
                        sim = similarity_matrix[idx1][idx2]
                        rec_similarities.append(sim)
            
            if rec_similarities:
                # Diversity is inverse of average similarity
                avg_similarity = np.mean(rec_similarities)
                diversity_scores.append(1 - avg_similarity)
        
        return {
            'average_diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'diversity_std': np.std(diversity_scores) if diversity_scores else 0,
            'products_evaluated': len(diversity_scores)
        }
    
    def benchmark_performance(self, func, *args, **kwargs):
        """
        Benchmark the performance of a function.
        
        Args:
            func: Function to benchmark
            *args, **kwargs: Arguments to pass to the function
        
        Returns:
            tuple: (result, execution_time)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    def generate_evaluation_report(self, recommender_name, recommendations, 
                                 similarity_matrix=None, product_map=None, 
                                 total_products=None, execution_time=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            recommender_name: Name of the recommendation system
            recommendations: Generated recommendations
            similarity_matrix: Optional similarity matrix
            product_map: Optional product mapping
            total_products: Total number of products
            execution_time: Optional execution time
        
        Returns:
            dict: Comprehensive evaluation report
        """
        report = {
            'recommender': recommender_name,
            'timestamp': pd.Timestamp.now(),
            'total_recommendations': sum(len(recs) for recs in recommendations.values()),
            'products_evaluated': len(recommendations)
        }
        
        if execution_time:
            report['execution_time_seconds'] = execution_time
            report['recommendations_per_second'] = len(recommendations) / execution_time
        
        if total_products:
            coverage = self.evaluate_coverage(recommendations, total_products)
            report['coverage'] = coverage
        
        if similarity_matrix is not None and product_map is not None:
            diversity = self.calculate_diversity(recommendations, similarity_matrix, product_map)
            report['diversity'] = diversity
            
            similarity_stats = self.calculate_similarity_stats(similarity_matrix)
            report['similarity_stats'] = similarity_stats
        
        return report
    
    def print_evaluation_report(self, report):
        """
        Print a formatted evaluation report.
        
        Args:
            report: Evaluation report dictionary
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {report['recommender']}")
        print(f"{'='*60}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Products Evaluated: {report['products_evaluated']}")
        print(f"Total Recommendations: {report['total_recommendations']}")
        
        if 'execution_time_seconds' in report:
            print(f"Execution Time: {report['execution_time_seconds']:.2f} seconds")
            print(f"Recommendations/Second: {report['recommendations_per_second']:.2f}")
        
        if 'coverage' in report:
            cov = report['coverage']
            print(f"\nCOVERAGE METRICS:")
            print(f"  Catalog Coverage: {cov['catalog_coverage']:.2f}%")
            print(f"  Prediction Coverage: {cov['prediction_coverage']:.2f}%")
            print(f"  Unique Recommended Products: {cov['unique_recommended_products']}")
        
        if 'diversity' in report:
            div = report['diversity']
            print(f"\nDIVERSITY METRICS:")
            print(f"  Average Diversity Score: {div['average_diversity']:.4f}")
            print(f"  Diversity Standard Deviation: {div['diversity_std']:.4f}")
        
        if 'similarity_stats' in report:
            sim = report['similarity_stats']
            print(f"\nSIMILARITY MATRIX STATS:")
            print(f"  Non-zero Similarities: {sim['non_zero_similarities']} / {sim['total_similarities']} ({sim['non_zero_percentage']:.2f}%)")
            print(f"  Mean Similarity: {sim['mean_similarity']:.4f}")
            print(f"  Max Similarity: {sim['max_similarity']:.4f}")
        
        print(f"{'='*60}\n")


def compare_recommenders(recommenders_results):
    """
    Compare multiple recommendation systems.
    
    Args:
        recommenders_results: Dict of {name: evaluation_report}
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for name, report in recommenders_results.items():
        row = {
            'Recommender': name,
            'Products_Evaluated': report['products_evaluated'],
            'Total_Recommendations': report['total_recommendations'],
        }
        
        if 'execution_time_seconds' in report:
            row['Execution_Time_Sec'] = report['execution_time_seconds']
            row['Recs_Per_Second'] = report['recommendations_per_second']
        
        if 'coverage' in report:
            row['Catalog_Coverage_%'] = report['coverage']['catalog_coverage']
            row['Prediction_Coverage_%'] = report['coverage']['prediction_coverage']
        
        if 'diversity' in report:
            row['Avg_Diversity'] = report['diversity']['average_diversity']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)
