# Hybrid Recommendation Engine

## Overview

This approach **combines collaborative filtering and content-based methods** using matrix factorization (NMF) and content similarity. It provides the best of both worlds by leveraging user interaction data and product content features for maximum accuracy and coverage.

## How It Works

1. **Data Collection**: Fetches detailed interaction data with vendor-product mappings
2. **Matrix Creation**: Creates product-user interaction matrix
3. **NMF Decomposition**: Uses Non-negative Matrix Factorization for feature extraction
4. **Content Similarity**: Calculates content-based similarity using master product groupings
5. **Hybrid Combination**: Combines collaborative and content similarities with weights
6. **Recommendation Generation**: Generates recommendations using the combined similarity matrix

## Key Features

- ✅ Combines strengths of collaborative and content-based filtering
- ✅ Uses advanced matrix factorization (NMF)
- ✅ Handles both cold start and data sparsity issues
- ✅ Configurable weighting between approaches
- ✅ Model persistence (save/load functionality)
- ✅ Best overall accuracy and coverage
- ❌ More computationally intensive
- ❌ Requires both user and content data

## Algorithm Components

### 1. Collaborative Filtering (70% weight)
- Uses NMF to extract latent features from user-item interactions
- Calculates cosine similarity on NMF feature space
- Handles sparse interaction data effectively

### 2. Content-Based Filtering (30% weight)
- Groups products by master product categories
- Assigns high similarity to products in same master category
- Provides recommendations for new products

### 3. Hybrid Combination
```python
hybrid_similarity = 0.7 * collaborative_sim + 0.3 * content_sim
```

## Usage

```python
from approach_3_hybrid.hybrid_recommender import HybridRecommender

# Initialize and train
recommender = HybridRecommender()
recommender.fit()

# Get recommendations for a product
recommendations = recommender.get_recommendations(product_id=123, top_n=5)

# Generate recommendations for all products
all_recs = recommender.generate_all_recommendations()

# Save to database
recommender.save_to_database()

# Save/load model
recommender.save_model("hybrid_model.pkl")
recommender.load_model("hybrid_model.pkl")
```

## Configuration

Adjust parameters in `config/database.py`:

```python
RECOMMENDATION_CONFIG = {
    "matrix_factorization_components": 50,  # NMF components
    "similarity_threshold": 0.01,           # Minimum similarity score
    "top_n_recommendations": 5,             # Number of recommendations
    "min_interactions": 1                   # Minimum interactions required
}
```

You can also adjust hybrid weights:

```python
recommender = HybridRecommender()
recommender.collaborative_weight = 0.8  # 80% collaborative
recommender.content_weight = 0.2        # 20% content-based
```

## Database Tables Used

- `po_details` - Purchase order details
- `po_items` - Purchase order items  
- `vendor_products` - Product and vendor information
- `recommended_categories` - Output recommendations

## Performance

- **Best for**: Maximum accuracy and coverage across all scenarios
- **Time Complexity**: O(n²) + O(k×n×m) where k=NMF components
- **Space Complexity**: O(n²) for similarity matrices
- **Typical Runtime**: ~60-120 seconds for 1000+ products

## Model Persistence

The hybrid recommender supports saving and loading trained models:

```python
# Save trained model
recommender.save_model("models/hybrid_recommender.pkl")

# Load pre-trained model
new_recommender = HybridRecommender()
new_recommender.load_model("models/hybrid_recommender.pkl")
```

## Advanced Features

### Matrix Factorization
- Uses scikit-learn's NMF implementation
- Automatically adjusts components based on data size
- Fallback to basic collaborative filtering if NMF fails

### Weighted Combination
- Configurable weights for different approaches
- Default: 70% collaborative, 30% content-based
- Can be tuned based on data characteristics

### Error Handling
- Graceful degradation when components fail
- Comprehensive logging and statistics
- Robust handling of sparse data
