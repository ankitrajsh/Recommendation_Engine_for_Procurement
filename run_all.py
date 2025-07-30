#!/usr/bin/env python3
"""
Master script to run all recommendation engine approaches.
"""

import sys
import os
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.demo import RecommendationDemo
from examples.comparison import RecommendationComparison
from data.sample_data import SampleDataGenerator


def run_demo():
    """Run the interactive demo."""
    print("Starting Interactive Demo...")
    demo = RecommendationDemo()
    demo.initialize_recommenders()
    demo.interactive_demo()


def run_comparison():
    """Run the comprehensive comparison."""
    print("Starting Comprehensive Comparison...")
    comparison = RecommendationComparison()
    comparison.run_full_comparison()


def run_collaborative():
    """Run only the collaborative filtering approach."""
    from approach_1_collaborative.collaborative_recommender import main as collaborative_main
    print("Running Collaborative Filtering Approach...")
    collaborative_main()


def run_content_based():
    """Run only the content-based filtering approach."""
    from approach_2_content_based.content_based_recommender import main as content_main
    print("Running Content-Based Filtering Approach...")
    content_main()


def run_hybrid():
    """Run only the hybrid approach."""
    from approach_3_hybrid.hybrid_recommender import main as hybrid_main
    print("Running Hybrid Approach...")
    hybrid_main()


def generate_sample_data():
    """Generate sample data for testing."""
    print("Generating Sample Data...")
    generator = SampleDataGenerator()
    datasets = generator.generate_complete_sample_dataset()
    generator.save_sample_data(datasets)
    print("Sample data generated successfully!")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Recommendation Engine Workspace')
    parser.add_argument('command', choices=[
        'demo', 'comparison', 'collaborative', 'content', 'hybrid', 'sample-data', 'all'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    print("RECOMMENDATION ENGINE WORKSPACE")
    print("=" * 50)
    
    if args.command == 'demo':
        run_demo()
    elif args.command == 'comparison':
        run_comparison()
    elif args.command == 'collaborative':
        run_collaborative()
    elif args.command == 'content':
        run_content_based()
    elif args.command == 'hybrid':
        run_hybrid()
    elif args.command == 'sample-data':
        generate_sample_data()
    elif args.command == 'all':
        print("Running all approaches sequentially...")
        print("\n1. Collaborative Filtering:")
        run_collaborative()
        print("\n2. Content-Based Filtering:")
        run_content_based()
        print("\n3. Hybrid Approach:")
        run_hybrid()
        print("\n4. Comprehensive Comparison:")
        run_comparison()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("RECOMMENDATION ENGINE WORKSPACE")
        print("=" * 50)
        print("Usage: python run_all.py <command>")
        print("\nAvailable commands:")
        print("  demo         - Run interactive demo")
        print("  comparison   - Run comprehensive comparison")
        print("  collaborative- Run collaborative filtering only")
        print("  content      - Run content-based filtering only")
        print("  hybrid       - Run hybrid approach only")
        print("  sample-data  - Generate sample data")
        print("  all          - Run all approaches")
        print("\nExamples:")
        print("  python run_all.py demo")
        print("  python run_all.py comparison")
        print("  python run_all.py all")
    else:
        main()
