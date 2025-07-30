"""
Interactive demo for all three recommendation engine approaches.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approach_1_collaborative.collaborative_recommender import CollaborativeRecommender
from approach_2_content_based.content_based_recommender import ContentBasedRecommender
from approach_3_hybrid.hybrid_recommender import HybridRecommender


class RecommendationDemo:
    """Interactive demo for recommendation engines."""
    
    def __init__(self):
        self.recommenders = {}
        self.trained = {}
    
    def initialize_recommenders(self):
        """Initialize all three recommendation approaches."""
        print("Initializing recommendation engines...")
        print("=" * 50)
        
        self.recommenders = {
            'collaborative': CollaborativeRecommender(),
            'content_based': ContentBasedRecommender(),
            'hybrid': HybridRecommender()
        }
        
        self.trained = {name: False for name in self.recommenders.keys()}
        print("All recommenders initialized!")
    
    def train_recommender(self, approach):
        """Train a specific recommender."""
        if approach not in self.recommenders:
            print(f"Unknown approach: {approach}")
            return False
        
        if self.trained[approach]:
            print(f"{approach.title()} recommender already trained!")
            return True
        
        print(f"\nTraining {approach.title()} Recommender...")
        print("-" * 40)
        
        try:
            self.recommenders[approach].fit()
            self.trained[approach] = True
            print(f"{approach.title()} recommender trained successfully!")
            return True
        except Exception as e:
            print(f"Error training {approach} recommender: {e}")
            return False
    
    def train_all_recommenders(self):
        """Train all recommendation approaches."""
        print("\nTraining all recommendation engines...")
        print("=" * 50)
        
        for approach in self.recommenders.keys():
            success = self.train_recommender(approach)
            if not success:
                print(f"Failed to train {approach} recommender")
            print()
    
    def get_recommendations_demo(self, approach, **kwargs):
        """Get recommendations from a specific approach."""
        if approach not in self.recommenders:
            print(f"Unknown approach: {approach}")
            return []
        
        if not self.trained[approach]:
            print(f"{approach.title()} recommender not trained. Training now...")
            if not self.train_recommender(approach):
                return []
        
        try:
            if approach == 'content_based':
                # Content-based requires vendor and product name
                vendor = kwargs.get('vendor_name')
                product = kwargs.get('product_name')
                if not vendor or not product:
                    print("Content-based recommender requires vendor_name and product_name")
                    return []
                return self.recommenders[approach].get_recommendations(vendor, product)
            else:
                # Collaborative and hybrid require product_id
                product_id = kwargs.get('product_id')
                if product_id is None:
                    print(f"{approach.title()} recommender requires product_id")
                    return []
                return self.recommenders[approach].get_recommendations(product_id)
        
        except Exception as e:
            print(f"Error getting recommendations from {approach}: {e}")
            return []
    
    def compare_approaches(self, **kwargs):
        """Compare recommendations from all approaches."""
        print("\nComparing All Approaches")
        print("=" * 40)
        
        results = {}
        
        # Get recommendations from each approach
        for approach in self.recommenders.keys():
            print(f"\n{approach.title()} Recommendations:")
            recommendations = self.get_recommendations_demo(approach, **kwargs)
            results[approach] = recommendations
            
            if recommendations:
                if approach == 'content_based':
                    for i, rec in enumerate(recommendations[:5], 1):
                        print(f"  {i}. {rec['message']}")
                else:
                    for i, (rec_id, score) in enumerate(recommendations[:5], 1):
                        print(f"  {i}. Product {rec_id} (Score: {score:.4f})")
            else:
                print("  No recommendations found")
        
        return results
    
    def interactive_demo(self):
        """Run interactive demo."""
        print("\n" + "=" * 60)
        print("RECOMMENDATION ENGINE INTERACTIVE DEMO")
        print("=" * 60)
        
        while True:
            print("\nAvailable options:")
            print("1. Train all recommenders")
            print("2. Train specific recommender")
            print("3. Get collaborative filtering recommendations")
            print("4. Get content-based recommendations")
            print("5. Get hybrid recommendations")
            print("6. Compare all approaches")
            print("7. Show training status")
            print("8. Exit")
            
            try:
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == '1':
                    self.train_all_recommenders()
                
                elif choice == '2':
                    print("\nAvailable approaches:")
                    for i, approach in enumerate(self.recommenders.keys(), 1):
                        status = "✓ Trained" if self.trained[approach] else "✗ Not trained"
                        print(f"  {i}. {approach.title()} ({status})")
                    
                    approach_choice = input("Enter approach number: ").strip()
                    approaches = list(self.recommenders.keys())
                    
                    if approach_choice.isdigit() and 1 <= int(approach_choice) <= len(approaches):
                        approach = approaches[int(approach_choice) - 1]
                        self.train_recommender(approach)
                    else:
                        print("Invalid choice")
                
                elif choice == '3':
                    product_id = input("Enter product ID: ").strip()
                    if product_id.isdigit():
                        recs = self.get_recommendations_demo('collaborative', product_id=int(product_id))
                        if recs:
                            print(f"\nCollaborative recommendations for Product {product_id}:")
                            for i, (rec_id, score) in enumerate(recs[:5], 1):
                                print(f"  {i}. Product {rec_id} (Score: {score:.4f})")
                    else:
                        print("Invalid product ID")
                
                elif choice == '4':
                    vendor = input("Enter vendor name: ").strip()
                    product = input("Enter product name: ").strip()
                    recs = self.get_recommendations_demo('content_based', 
                                                       vendor_name=vendor, product_name=product)
                    if recs:
                        print(f"\nContent-based recommendations for '{product}' by '{vendor}':")
                        for i, rec in enumerate(recs[:5], 1):
                            print(f"  {i}. {rec['message']}")
                
                elif choice == '5':
                    product_id = input("Enter product ID: ").strip()
                    if product_id.isdigit():
                        recs = self.get_recommendations_demo('hybrid', product_id=int(product_id))
                        if recs:
                            print(f"\nHybrid recommendations for Product {product_id}:")
                            for i, (rec_id, score) in enumerate(recs[:5], 1):
                                print(f"  {i}. Product {rec_id} (Score: {score:.4f})")
                    else:
                        print("Invalid product ID")
                
                elif choice == '6':
                    print("\nChoose comparison type:")
                    print("1. Compare by product ID (collaborative + hybrid)")
                    print("2. Compare by vendor/product (content-based only)")
                    
                    comp_choice = input("Enter choice (1-2): ").strip()
                    
                    if comp_choice == '1':
                        product_id = input("Enter product ID: ").strip()
                        if product_id.isdigit():
                            self.compare_approaches(product_id=int(product_id))
                    elif comp_choice == '2':
                        vendor = input("Enter vendor name: ").strip()
                        product = input("Enter product name: ").strip()
                        self.compare_approaches(vendor_name=vendor, product_name=product)
                
                elif choice == '7':
                    print("\nTraining Status:")
                    for approach, trained in self.trained.items():
                        status = "✓ Trained" if trained else "✗ Not trained"
                        print(f"  {approach.title()}: {status}")
                
                elif choice == '8':
                    print("Exiting demo. Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-8.")
            
            except KeyboardInterrupt:
                print("\n\nExiting demo. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


def main():
    """Main function to run the demo."""
    demo = RecommendationDemo()
    demo.initialize_recommenders()
    demo.interactive_demo()


if __name__ == "__main__":
    main()
