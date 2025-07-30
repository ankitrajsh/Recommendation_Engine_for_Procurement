"""
Database configuration for the recommendation engine workspace.
"""

# Database connection configuration
DB_CONFIG = {
    "host": "34.93.93.62",
    "database": "bluet_devpy_final_stg",
    "user": "postgres",
    "password": "Y5mnshpDFF44",
    "port": "5432"
}

# Analytics database configuration (used by content-based recommender)
ANALYTICS_DB_CONFIG = {
    "host": "34.93.93.62",
    "database": "analytics_database",
    "user": "postgres",
    "password": "Y5mnshpDFF44",
    "port": "5432"
}

# Recommendation system parameters
RECOMMENDATION_CONFIG = {
    "min_interactions": 1,
    "similarity_threshold": 0.01,
    "top_n_recommendations": 5,
    "co_occurrence_threshold": 0.8,
    "matrix_factorization_components": 50
}
