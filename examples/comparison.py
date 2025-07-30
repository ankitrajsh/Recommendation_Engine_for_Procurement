"""
Comprehensive comparison of all three recommendation engine approaches.
"""

import sys
import os
import time
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approach_1_collaborative.collaborative_recommender import CollaborativeRecommender
from approach_2_content_based.content_based_recommender import ContentBasedRecommender
from approach_3_hybrid.hybrid_recommender import HybridRecommender
from utils.evaluation import RecommendationEvaluator, compare_recommenders


class RecommendationComparison:
    """Comprehensive comparison of recommendation approaches."""
    
    def __init__(self):
        self.recommenders = {}
        self.evaluation_reports = {}
        self.evaluator = RecommendationEvaluator()
    
    def initialize_recommenders(self):
        """Initialize all recommendation approaches."""
        print("Initializing Recommendation Engines")
        print("=" * 50)
        
        self.recommenders = {
            'Collaborative Filtering': CollaborativeRecommender(),
            'Content-Based Filtering': ContentBasedRecommender(),
            'Hybrid Approach': HybridRecommender()
        }
        
        print("All recommenders initialized successfully!")
    
    def train_and_evaluate_all(self):
        """Train and evaluate all recommendation approaches."""
        print("\nTraining and Evaluating All Approaches")
        print("=" * 60)
        
        for name, recommender in self.recommenders.items():
            print(f"\n{'='*20} {name} {'='*20}")
            
            try:
                # Train the recommender and measure time
                start_time = time.time()
                recommender.fit()
                training_time = time.time() - start_time
                
                # Generate recommendations and measure time
                start_time = time.time()
                recommendations = recommender.generate_all_recommendations()
                recommendation_time = time.time() - start_time
                
                total_time = training_time + recommendation_time
                
                # Evaluate the recommender
                if hasattr(recommender, 'evaluate'):
                    evaluation_report = recommender.evaluate(recommendations)
                    evaluation_report['training_time_seconds'] = training_time
                    evaluation_report['recommendation_time_seconds'] = recommendation_time
                    evaluation_report['total_time_seconds'] = total_time
                else:
                    # Fallback evaluation for content-based
                    evaluation_report = self.evaluator.generate_evaluation_report(
                        name, recommendations, execution_time=total_time
                    )
                    evaluation_report['training_time_seconds'] = training_time
                    evaluation_report['recommendation_time_seconds'] = recommendation_time
                
                self.evaluation_reports[name] = evaluation_report
                
                # Print individual report
                self.evaluator.print_evaluation_report(evaluation_report)
                
                print(f"Training Time: {training_time:.2f} seconds")
                print(f"Recommendation Time: {recommendation_time:.2f} seconds")
                print(f"Total Time: {total_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                import traceback
                traceback.print_exc()
    
    def generate_comparison_table(self):
        """Generate a comparison table of all approaches."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON TABLE")
        print("=" * 80)
        
        if not self.evaluation_reports:
            print("No evaluation reports available. Run train_and_evaluate_all() first.")
            return None
        
        # Create comparison DataFrame
        comparison_df = compare_recommenders(self.evaluation_reports)
        
        # Add custom metrics
        for name, report in self.evaluation_reports.items():
            idx = comparison_df[comparison_df['Recommender'] == name].index
            if len(idx) > 0:
                idx = idx[0]
                comparison_df.loc[idx, 'Training_Time_Sec'] = report.get('training_time_seconds', 0)
                comparison_df.loc[idx, 'Rec_Time_Sec'] = report.get('recommendation_time_seconds', 0)
        
        # Display the table
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return comparison_df
    
    def analyze_strengths_weaknesses(self):
        """Analyze strengths and weaknesses of each approach."""
        print("\n" + "=" * 80)
        print("STRENGTHS & WEAKNESSES ANALYSIS")
        print("=" * 80)
        
        analysis = {
            'Collaborative Filtering': {
                'strengths': [
                    'Works well with sparse interaction data',
                    'No product content/metadata required',
                    'Finds products purchased by similar users',
                    'Scales well with large user bases'
                ],
                'weaknesses': [
                    'Cold start problem for new products',
                    'Requires sufficient user interaction data',
                    'May not work well for niche products'
                ],
                'best_for': 'Products with rich user interaction history'
            },
            'Content-Based Filtering': {
                'strengths': [
                    'Works well for new products (no cold start)',
                    'Fast dictionary-based lookups',
                    'No user interaction data required',
                    'Explainable recommendations'
                ],
                'weaknesses': [
                    'Requires good vendor catalog data',
                    'Limited to vendor overlap patterns',
                    'May miss user preference patterns'
                ],
                'best_for': 'New products or when user data is limited'
            },
            'Hybrid Approach': {
                'strengths': [
                    'Combines strengths of both approaches',
                    'Uses advanced matrix factorization',
                    'Handles both cold start and sparsity',
                    'Best overall accuracy and coverage'
                ],
                'weaknesses': [
                    'More computationally intensive',
                    'Requires both user and content data',
                    'More complex to tune and maintain'
                ],
                'best_for': 'Maximum accuracy and coverage across all scenarios'
            }
        }
        
        for approach, details in analysis.items():
            print(f"\n{approach}:")
            print(f"  Strengths:")
            for strength in details['strengths']:
                print(f"    ✅ {strength}")
            print(f"  Weaknesses:")
            for weakness in details['weaknesses']:
                print(f"    ❌ {weakness}")
            print(f"  Best for: {details['best_for']}")
    
    def recommend_best_approach(self):
        """Recommend the best approach based on evaluation results."""
        print("\n" + "=" * 80)
        print("RECOMMENDATION: WHICH APPROACH TO USE?")
        print("=" * 80)
        
        if not self.evaluation_reports:
            print("No evaluation data available.")
            return
        
        # Analyze metrics to recommend best approach
        recommendations = []
        
        # Check coverage
        best_coverage = max(
            (name, report.get('coverage', {}).get('catalog_coverage', 0))
            for name, report in self.evaluation_reports.items()
            if 'coverage' in report
        )
        
        # Check speed
        best_speed = min(
            (name, report.get('total_time_seconds', float('inf')))
            for name, report in self.evaluation_reports.items()
        )
        
        # Check diversity
        best_diversity = max(
            (name, report.get('diversity', {}).get('average_diversity', 0))
            for name, report in self.evaluation_reports.items()
            if 'diversity' in report
        )
        
        print("📊 PERFORMANCE ANALYSIS:")
        print(f"  🎯 Best Coverage: {best_coverage[0]} ({best_coverage[1]:.1f}%)")
        print(f"  ⚡ Fastest: {best_speed[0]} ({best_speed[1]:.1f}s)")
        if best_diversity[1] > 0:
            print(f"  🌟 Most Diverse: {best_diversity[0]} ({best_diversity[1]:.3f})")
        
        print("\n🎯 RECOMMENDATIONS:")
        
        print("\n1. FOR PRODUCTION USE:")
        print("   → Hybrid Approach - Best overall performance and coverage")
        
        print("\n2. FOR FAST PROTOTYPING:")
        print("   → Content-Based Filtering - Quick setup, no user data needed")
        
        print("\n3. FOR USER-RICH ENVIRONMENTS:")
        print("   → Collaborative Filtering - Leverages user behavior patterns")
        
        print("\n4. FOR NEW PRODUCT LAUNCHES:")
        print("   → Content-Based or Hybrid - Handles cold start problem")
        
        print("\n💡 IMPLEMENTATION STRATEGY:")
        print("   1. Start with Content-Based for immediate results")
        print("   2. Add Collaborative as user data grows")
        print("   3. Implement Hybrid for maximum performance")
        print("   4. A/B test different approaches with real users")
    
    def save_comparison_results(self, output_dir='comparison_results'):
        """Save comparison results to files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison table
        if self.evaluation_reports:
            comparison_df = compare_recommenders(self.evaluation_reports)
            comparison_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
            print(f"\nComparison table saved to {output_dir}/comparison_table.csv")
        
        # Save detailed reports
        for name, report in self.evaluation_reports.items():
            filename = name.lower().replace(' ', '_') + '_report.txt'
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"Evaluation Report: {name}\n")
                f.write("=" * 50 + "\n")
                for key, value in report.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Detailed report saved to {filepath}")
    
    def run_full_comparison(self):
        """Run the complete comparison analysis."""
        print("RECOMMENDATION ENGINE COMPARISON SUITE")
        print("=" * 60)
        
        # Initialize recommenders
        self.initialize_recommenders()
        
        # Train and evaluate all approaches
        self.train_and_evaluate_all()
        
        # Generate comparison table
        self.generate_comparison_table()
        
        # Analyze strengths and weaknesses
        self.analyze_strengths_weaknesses()
        
        # Recommend best approach
        self.recommend_best_approach()
        
        # Save results
        self.save_comparison_results()
        
        print("\n" + "=" * 60)
        print("COMPARISON ANALYSIS COMPLETED!")
        print("=" * 60)


def main():
    """Main function to run the comparison."""
    comparison = RecommendationComparison()
    comparison.run_full_comparison()


if __name__ == "__main__":
    main()
