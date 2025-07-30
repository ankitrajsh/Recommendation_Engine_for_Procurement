# Recommendation Engine Workspace

This workspace contains three different recommendation engine implementations for product recommendations.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run any of the three approaches
python approach_1_collaborative/collaborative_recommender.py
python approach_2_content_based/content_based_recommender.py  
python approach_3_hybrid/hybrid_recommender.py
```

## 📁 Workspace Structure

```
VIPANI/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── config/
│   ├── __init__.py
│   └── database.py                     # Database configuration
├── data/
│   ├── __init__.py
│   └── sample_data.py                  # Sample data generation
├── utils/
│   ├── __init__.py
│   ├── database_utils.py               # Database utilities
│   └── evaluation.py                  # Model evaluation utilities
├── approach_1_collaborative/
│   ├── __init__.py
│   ├── collaborative_recommender.py   # User-item collaborative filtering
│   └── README.md
├── approach_2_content_based/
│   ├── __init__.py
│   ├── content_based_recommender.py   # Content-based filtering
│   └── README.md
├── approach_3_hybrid/
│   ├── __init__.py
│   ├── hybrid_recommender.py          # Hybrid approach
│   └── README.md
└── examples/
    ├── __init__.py
    ├── demo.py                         # Interactive demo
    └── comparison.py                   # Compare all approaches
```

## 🔧 Three Recommendation Approaches

### 1. Collaborative Filtering (approach_1_collaborative/)
- **Method**: User-item interaction matrix with cosine similarity
- **Best for**: Products with rich user interaction data
- **Key Features**: 
  - Finds products purchased by similar users
  - Works well with sparse data
  - No product content needed

### 2. Content-Based Filtering (approach_2_content_based/)
- **Method**: Co-occurrence analysis based on vendor catalogs
- **Best for**: New products or when user data is limited
- **Key Features**:
  - Analyzes vendor product catalogs
  - Finds products commonly sold together
  - Fast dictionary-based lookups

### 3. Hybrid Approach (approach_3_hybrid/)
- **Method**: Combines collaborative and content-based methods
- **Best for**: Maximum accuracy and coverage
- **Key Features**:
  - Matrix factorization (NMF)
  - Content similarity using TF-IDF
  - Weighted combination of approaches

## 🗄️ Database Setup

The system connects to PostgreSQL database with the following tables:
- `po_details` - Purchase order details
- `po_items` - Purchase order items  
- `vendor_products` - Vendor product catalog
- `recommended_categories` - Generated recommendations

## 📊 Usage Examples

```python
# Collaborative Filtering
from approach_1_collaborative.collaborative_recommender import CollaborativeRecommender
recommender = CollaborativeRecommender()
recommendations = recommender.get_recommendations(product_id=123)

# Content-Based Filtering  
from approach_2_content_based.content_based_recommender import ContentBasedRecommender
recommender = ContentBasedRecommender()
recommendations = recommender.get_recommendations(vendor="ABC Corp", product="Widget")

# Hybrid Approach
from approach_3_hybrid.hybrid_recommender import HybridRecommender
recommender = HybridRecommender()
recommendations = recommender.get_recommendations(product_id=123)
```

## 🔍 Evaluation & Comparison

Run the comparison script to evaluate all three approaches:

```bash
python examples/comparison.py
```

This will show:
- Recommendation accuracy metrics
- Performance benchmarks
- Coverage analysis
- Recommendation diversity

## 🛠️ Configuration

Update database credentials in `config/database.py`:

```python
DB_CONFIG = {
    "host": "your-host",
    "database": "your-database", 
    "user": "your-username",
    "password": "your-password",
    "port": "5432"
}
```

## 📈 Performance Tips

1. **Collaborative Filtering**: Works best with >100 user interactions per product
2. **Content-Based**: Requires good product metadata and vendor catalogs
3. **Hybrid**: Combines strengths but requires more computational resources

## 🤝 Contributing

1. Each approach is self-contained in its own directory
2. Shared utilities are in the `utils/` directory
3. Add new approaches by creating a new `approach_X_name/` directory
4. Follow the same interface pattern for consistency
