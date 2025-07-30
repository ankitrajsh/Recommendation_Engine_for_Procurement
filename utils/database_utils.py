"""
Database utility functions for the recommendation engine workspace.
"""

import psycopg2
import pandas as pd
from config.database import DB_CONFIG, ANALYTICS_DB_CONFIG


def get_database_connection(use_analytics=False):
    """
    Create and return a database connection.
    
    Args:
        use_analytics (bool): If True, use analytics database config
    
    Returns:
        psycopg2.connection: Database connection object
    """
    config = ANALYTICS_DB_CONFIG if use_analytics else DB_CONFIG
    return psycopg2.connect(**config)


def fetch_purchase_data():
    """
    Fetch purchase interaction data from the database.
    
    Returns:
        tuple: (interactions_df, dates_df) - DataFrames with purchase data
    """
    conn = get_database_connection()
    
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
    
    df = pd.read_sql(query, conn)
    
    # Get dates for each product
    dates_df = df.groupby('product_id').agg({
        'created_date': 'max',
        'updated_date': 'max'
    }).reset_index()
    
    conn.close()
    return df, dates_df


def fetch_vendor_product_data():
    """
    Fetch vendor-product catalog data.
    
    Returns:
        pd.DataFrame: DataFrame with vendor and product information
    """
    conn = get_database_connection(use_analytics=True)
    
    query = """
        SELECT vendor_name, product_name
        FROM vendor_products_catalog
        WHERE vendor_name IS NOT NULL 
        AND product_name IS NOT NULL
        GROUP BY vendor_name, product_name
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def fetch_detailed_interaction_data():
    """
    Fetch detailed interaction data for hybrid recommendations.
    
    Returns:
        tuple: (interactions_df, product_df, vendor_to_master_map)
    """
    conn = get_database_connection()
    cursor = conn.cursor()

    # Fetch product interaction data
    cursor.execute("""
        SELECT 
            vp.product_id AS master_product_id,
            vp.id AS vendor_product_id,
            pd.buyer_user_id AS user_id,
            pd.total_qty AS interaction_strength
        FROM po_details pd
        JOIN po_items pi ON pd.id = pi.po_id
        JOIN vendor_products vp ON pi.product_id = vp.id
        GROUP BY vp.product_id, vp.id, pd.buyer_user_id, pd.total_qty
    """)
    
    interactions_data = cursor.fetchall()
    interactions_df = pd.DataFrame(interactions_data, 
                                 columns=['master_product_id', 'vendor_product_id', 
                                         'user_id', 'interaction_strength'])

    # Create vendor to master product mapping
    vendor_to_master_map = (interactions_df[['vendor_product_id', 'master_product_id']]
                           .drop_duplicates()
                           .set_index('vendor_product_id')['master_product_id']
                           .to_dict())

    # Create product DataFrame
    unique_products = interactions_df['master_product_id'].unique()
    product_df = pd.DataFrame({'product_id': unique_products})

    cursor.close()
    conn.close()
    
    return interactions_df, product_df, vendor_to_master_map


def get_category_data(product_ids):
    """
    Get category data for a list of products.
    
    Args:
        product_ids (list): List of product IDs
    
    Returns:
        dict: Mapping of product_id to category_ids
    """
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT product_id, category_ids FROM vendor_products WHERE product_id = ANY(%s)",
        (product_ids,)
    )
    
    result = dict(cursor.fetchall())
    cursor.close()
    conn.close()
    
    return result


def store_recommendations(recommendations, table_name="recommended_categories"):
    """
    Store recommendations in the database.
    
    Args:
        recommendations (list): List of recommendation tuples
        table_name (str): Name of the table to store recommendations
    """
    from psycopg2.extras import execute_batch
    
    conn = get_database_connection()
    cursor = conn.cursor()

    insert_query = f"""
        INSERT INTO {table_name} 
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
    
    cursor.close()
    conn.close()
    
    print(f"Stored {len(recommendations)} recommendations in {table_name}")
