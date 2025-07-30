# Content-Based Filtering Recommendation Engine

## Overview

This approach uses **co-occurrence analysis** based on vendor catalogs to find products that are commonly sold together by the same vendors. It analyzes vendor product catalogs and recommends products based on vendor overlap patterns.

## How It Works

1. **Data Collection**: Fetches vendor-product catalog data from the database
2. **Dictionary Building**: Creates efficient lookup dictionaries for products and vendors
3. **Co-occurrence Analysis**: Calculates how often products appear together in vendor catalogs
4. **Recommendation Generation**: Recommends products with high co-occurrence percentages

## Key Features

- ✅ Works well for new products (no cold start problem)
- ✅ Fast dictionary-based lookups
- ✅ No user interaction data required
- ✅ Finds products commonly sold together
- ✅ Explainable recommendations
- ❌ Requires good vendor catalog data
- ❌ Limited to vendor overlap patterns

## Usage

```python
from approach_2_content_based.content_based_recommender import ContentBasedRecommender

# Initialize and train
recommender = ContentBasedRecommender()
recommender.fit()

# Get recommendations for a vendor-product combination
recommendations = recommender.get_recommendations(
    vendor_name="ABC Corp", 
    product_name="Widget A"
)

# Get recommendations for any product
recommendations = recommender.get_recommendations_by_product("Widget A")

# Generate recommendations for all products
all_recs = recommender.generate_all_recommendations()
```

## Configuration

Adjust parameters in `config/database.py`:

```python
RECOMMENDATION_CONFIG = {
    "co_occurrence_threshold": 0.8,  # Minimum co-occurrence percentage
    "top_n_recommendations": 5,      # Number of recommendations per product
}
```

## Database Tables Used

- `vendor_products_catalog` - Vendor product catalog data

## Algorithm Details

The system calculates co-occurrence as:
```
Co-occurrence(Product A, Product B) = 
    |Vendors selling both A and B| / |Vendors selling A|
```

Products are recommended if their co-occurrence percentage exceeds the threshold (default 80%).

## Performance

- **Best for**: New products or when user data is limited
- **Time Complexity**: O(n²) for co-occurrence calculation
- **Space Complexity**: O(n²) for co-occurrence storage
- **Typical Runtime**: ~10-30 seconds for 1000+ products

## Example Output

```
Recommendations for 'Laptop' by 'TechVendor':
1. 85.2% of vendors who have 'Laptop' also have 'Mouse'
2. 82.1% of vendors who have 'Laptop' also have 'Keyboard'
3. 80.5% of vendors who have 'Laptop' also have 'Monitor'
```
