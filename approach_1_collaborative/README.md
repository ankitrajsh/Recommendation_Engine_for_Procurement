# Collaborative Filtering Recommendation Engine

## Overview

This approach uses **user-item interaction data** to find products that are frequently purchased together by similar users. It builds a product-product similarity matrix based on cosine similarity of user purchase patterns.

## How It Works

1. **Data Collection**: Fetches purchase order data from the database
2. **Matrix Creation**: Creates a product-user interaction matrix with purchase quantities
3. **Similarity Calculation**: Uses cosine similarity to find similar products
4. **Recommendation Generation**: Recommends products with highest similarity scores

## Key Features

- ✅ Works well with sparse interaction data
- ✅ No product content/metadata required
- ✅ Finds products purchased by similar users
- ✅ Scales well with large user bases
- ❌ Cold start problem for new products
- ❌ Requires sufficient user interaction data

## Usage

```python
from approach_1_collaborative.collaborative_recommender import CollaborativeRecommender

# Initialize and train
recommender = CollaborativeRecommender()
recommender.fit()

# Get recommendations for a product
recommendations = recommender.get_recommendations(product_id=123, top_n=5)

# Generate recommendations for all products
all_recs = recommender.generate_all_recommendations()

# Save to database
recommender.save_to_database()
```

## Configuration

Adjust parameters in `config/database.py`:

```python
RECOMMENDATION_CONFIG = {
    "similarity_threshold": 0.01,    # Minimum similarity score
    "top_n_recommendations": 5,      # Number of recommendations per product
    "min_interactions": 1            # Minimum interactions required
}
```

## Database Tables Used

- `po_details` - Purchase order details
- `po_items` - Purchase order items  
- `vendor_products` - Product information
- `recommended_categories` - Output recommendations

## Performance

- **Best for**: Products with rich user interaction history
- **Time Complexity**: O(n²) for similarity calculation
- **Space Complexity**: O(n²) for similarity matrix storage
- **Typical Runtime**: ~30-60 seconds for 1000+ products
