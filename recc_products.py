import pandas as pd
import psycopg2
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from psycopg2.extras import execute_batch

# Database connection details
DB_CONFIG = {
    "host": "34.93.93.62",
    "database": "bluet_devpy_final_stg",
    "user": "postgres",
    "password": "Y5mnshpDFF44",
    "port": "5432"
}

def connect_to_db():
    """Simple function to connect to database"""
    return psycopg2.connect(**DB_CONFIG)

def get_category_level_4(category_json):
    """Extract level 4 category from JSON"""
    try:
        if isinstance(category_json, str):
            categories = json.loads(category_json)
        else:
            categories = category_json

        # Get level 4 category if it exists
        if 'cat_4' in categories:
            cat_4_list = categories['cat_4']
            if cat_4_list and len(cat_4_list) > 0:
                return str(cat_4_list[0].get('id'))
    except:
        pass
    return None

def fetch_purchase_data():
    """Fetch all purchase data from database"""
    print("Fetching purchase data...")

    conn = connect_to_db()
    cursor = conn.cursor()

    # Get purchase data with aggregated quantities
    query = """
        SELECT 
            vp.product_id,
            pd.buyer_user_id,
            SUM(pd.total_qty) as total_quantity,
            MAX(pi.created_date) as created_date,
            MAX(pi.updated_date) as updated_date,
            vp.category_ids
        FROM po_details pd
        JOIN po_items pi ON pd.id = pi.po_id
        JOIN vendor_products vp ON pi.product_id = vp.product_id
        WHERE pd.total_qty > 0
        GROUP BY vp.product_id, pd.buyer_user_id, vp.category_ids
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Create pandas DataFrame
    df = pd.DataFrame(rows, columns=[
        'product_id', 'user_id', 'quantity',
        'created_date', 'updated_date', 'category_ids'
    ])

    # Get dates for each product
    dates_by_product = df.groupby('product_id').agg({
        'created_date': 'max',
        'updated_date': 'max'
    }).reset_index()

    cursor.close()
    conn.close()

    return df, dates_by_product

def create_similarity_matrix(df):
    """Create product similarity matrix from purchase data"""
    print("Creating similarity matrix...")

    # Create product-user purchase matrix
    purchase_matrix = df.pivot_table(
        index='product_id',
        columns='user_id',
        values='quantity',
        fill_value=0
    )

    # Calculate similarity between products
    similarity = cosine_similarity(purchase_matrix)

    return similarity, purchase_matrix.index.tolist()

def get_category_data(product_ids, cursor):
    """Get category data for list of products"""
    cursor.execute(
        "SELECT product_id, category_ids FROM vendor_products WHERE product_id = ANY(%s)",
        (product_ids,)
    )
    return dict(cursor.fetchall())

def generate_recommendations():
    """Main function to generate product recommendations"""
    print("Starting recommendation generation...")

    # Get purchase data
    purchase_df, dates_df = fetch_purchase_data()
    print(f"Found {len(purchase_df)} purchase records")

    # Generate similarity matrix
    similarity_matrix, product_ids = create_similarity_matrix(purchase_df)
    print(f"Generated similarities for {len(product_ids)} products")

    # Connect to database for inserting recommendations
    conn = connect_to_db()
    cursor = conn.cursor()

    recommendations = []

    # Generate recommendations for each product
    for idx, current_product in enumerate(product_ids):
        # Get 5 most similar products
        similar_scores = similarity_matrix[idx]
        similar_indices = np.argsort(-similar_scores)[1:6]
        similar_products = [int(product_ids[i]) for i in similar_indices]

        # Get dates for current product
        product_dates = dates_df[dates_df['product_id'] == current_product].iloc[0]

        # Get categories for similar products
        category_data = get_category_data(similar_products, cursor)

        # Create recommendation entries
        for similar_product in similar_products:
            cat_4 = get_category_level_4(category_data.get(similar_product, '{}'))
            if cat_4:
                recommendations.append((
                    int(current_product),          # id
                    cat_4,                         # category_id
                    int(similar_product),          # product_id
                    'recommended products',        # product_type
                    'A',                          # status
                    product_dates['created_date'],
                    product_dates['updated_date']
                ))

    print(f"Generated {len(recommendations)} recommendations")

    # Insert recommendations if we have any
    if recommendations:
        insert_query = """
            INSERT INTO recommended_categories 
            (id, category_id, product_id, product_type, status, created_date, updated_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                category_id = EXCLUDED.category_id,
                product_id = EXCLUDED.product_id,
                product_type = EXCLUDED.product_type,
                status = EXCLUDED.status,
                updated_date = EXCLUDED.updated_date
        """

        execute_batch(cursor, insert_query, recommendations, page_size=1000)
        conn.commit()
        print("Saved recommendations to database")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    generate_recommendations()
