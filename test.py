import pandas as pd
import psycopg2
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from datetime import datetime


def generate_data():
    # PostgreSQL connection
    pg_conn = psycopg2.connect(
        host="34.93.93.62",
        database="bluet_devpy_final_stg",
        user="postgres",
        password="Y5mnshpDFF44",
        port="5432"
    )
    pg_cur = pg_conn.cursor()

    # Fetch product interaction data
    pg_cur.execute("""
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
    interactions_data = pg_cur.fetchall()
    interactions_df = pd.DataFrame(interactions_data, columns=['master_product_id', 'vendor_product_id', 'user_id', 'interaction_strength'])

    # Create a mapping between vendor_product_id and master_product_id for later use
    vendor_to_master_map = interactions_df[['vendor_product_id', 'master_product_id']].drop_duplicates().set_index('vendor_product_id')['master_product_id'].to_dict()
    product_data = pg_cur.fetchall()
    product_df = pd.DataFrame(product_data, columns=['product_id', 'product_name', 'category_ids', 'attributes'])

    pg_cur.close()
    pg_conn.close()

    # We will use only the interaction data since we're removing the master_products part
    product_df = pd.DataFrame()

    # Convert interaction data to be our product data
    # Group by master_product_id to get unique products
    unique_products = interactions_df['master_product_id'].unique()
    product_df = pd.DataFrame({'product_id': unique_products})

    return interactions_df, product_df


def create_product_interaction_matrix(interactions_df):
    # Create product-user interaction matrix using vendor_product_id instead of master_product_id
    product_user_matrix = interactions_df.pivot_table(
        index='vendor_product_id', columns='user_id', values='interaction_strength', fill_value=0
    )
    return product_user_matrix


def evaluate_models(interactions_df, product_df):
    product_user_matrix = create_product_interaction_matrix(interactions_df)

    # Print statistics about the matrix sparsity
    total_cells = product_user_matrix.shape[0] * product_user_matrix.shape[1]
    non_zero_cells = (product_user_matrix > 0).sum().sum()
    sparsity = 100 * (1 - non_zero_cells / total_cells)
    print(f"Matrix shape: {product_user_matrix.shape}, Sparsity: {sparsity:.2f}%")
    print(f"Non-zero interactions: {non_zero_cells} out of {total_cells}")

    # Filter out products with too few interactions (rows with mostly zeros)
    row_sums = product_user_matrix.sum(axis=1)
    min_interactions = 1  # Products must have at least one interaction
    valid_products = row_sums[row_sums >= min_interactions].index
    print(f"Products with at least {min_interactions} interaction(s): {len(valid_products)} out of {len(product_user_matrix)}")

    # Keep only products with sufficient interactions
    product_user_matrix = product_user_matrix.loc[valid_products]

    # Since we're removing the master_products part, we'll use only collaborative filtering
    # based on user interactions, without content-based filtering
    common_products = valid_products

    # Create aligned product_map
    product_map = {product_id: idx for idx, product_id in enumerate(common_products)}

    # Calculate and save similarity matrix
    print("Calculating similarity matrix...")

    # Use only collaborative filtering similarity (based on user interactions)
    # since we're removing the master_products part
    similarity_matrix = cosine_similarity(product_user_matrix)
    np.save('similarity_matrix.npy', similarity_matrix)

    # Check the similarity matrix for zero values
    matrix_zeros = (similarity_matrix == 0).sum()
    total_pairs = similarity_matrix.shape[0] * similarity_matrix.shape[1]
    print(f"Similarity matrix: {matrix_zeros} zeros out of {total_pairs} pairs ({100*matrix_zeros/total_pairs:.2f}%)")

    # Train NMF on the interaction matrix for additional modeling
    n_components = min(50, product_user_matrix.shape[0] - 1, product_user_matrix.shape[1] - 1)
    if n_components > 0:
        nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
        product_factors = nmf.fit_transform(product_user_matrix)
        np.save('product_factors.npy', product_factors)

    # Save product_map
    with open('product_map.pkl', 'wb') as file:
        pickle.dump(product_map, file)

    print("Model components saved successfully.")
    return similarity_matrix, product_map


def recommend_products(product_id, similarity_matrix, product_map, top_n=5, min_score=0.01):
    """
    Recommend top-N similar products for a given product.
    Returns a list of tuples (recommended_product_id, similarity_score)

    Args:
        product_id: The product ID to get recommendations for
        similarity_matrix: The precomputed similarity matrix
        product_map: Mapping from product_id to matrix index
        top_n: Maximum number of recommendations to return
        min_score: Minimum similarity score to consider (filters out very low/zero scores)
    """
    reverse_product_map = {v: k for k, v in product_map.items()}
    if product_id not in product_map:
        print(f"Product {product_id} not found in the product map. Ensure it is part of the input data.")
        return []

    product_index = product_map[product_id]
    similarities = similarity_matrix[product_index]

    # Check if there are any non-zero similarities
    non_zero_count = np.sum(similarities > min_score)
    if non_zero_count == 0:
        print(f"Warning: Product {product_id} has no significant similarities with other products.")
        # In this case, we might want to use a fallback strategy like popularity
        return []

    # Get top-N similar product indices with their scores
    # Skip the first one (which is the product itself with similarity=1)
    similar_indices = np.argsort(-similarities)[1:top_n + 1]

    # Filter by minimum score and convert to list of tuples with actual product IDs
    recommended_products = []
    for idx in similar_indices:
        if idx in reverse_product_map and similarities[idx] > min_score:
            rec_id = reverse_product_map[idx]
            score = float(similarities[idx])
            recommended_products.append((rec_id, score))

    # Debug info
    if recommended_products:
        print(f"Found {len(recommended_products)} recommendations with scores above {min_score}")
    else:
        print(f"No recommendations with scores above {min_score}")

    return recommended_products


def store_recommendations_in_db(product_recommendations, vendor_to_master_map):
    """
    Store the generated product recommendations in the database.

    Args:
        product_recommendations: A dict with vendor_product_id as key and a list of
                               (recommended_vendor_product_id, score) tuples as value
        vendor_to_master_map: A dict mapping vendor_product_id to master_product_id
    """
    # PostgreSQL connection
    pg_conn = psycopg2.connect(
        host="34.93.93.62",
        database="bluet_devpy_final_stg",
        user="postgres",
        password="Y5mnshpDFF44",
        port="5432"
    )
    pg_cur = pg_conn.cursor()

    current_time = datetime.now()

    # First, delete existing entries for similar recommendation type
    pg_cur.execute("""
        DELETE FROM product_recommendations 
        WHERE recommendation_type = 'similar' AND target_type = 'product'
    """)

    # Prepare batch insert data
    insert_data = []
    for vendor_product_id, recommendations in product_recommendations.items():
        # Map vendor_product_id to master_product_id for target_id
        if vendor_product_id not in vendor_to_master_map:
            print(f"Warning: No master_product_id found for vendor_product_id {vendor_product_id}. Skipping.")
            continue

        master_product_id = vendor_to_master_map[vendor_product_id]

        for rank, (rec_vendor_product_id, score) in enumerate(recommendations, 1):
            # Map recommended vendor_product_id to master_product_id
            if rec_vendor_product_id not in vendor_to_master_map:
                print(f"Warning: No master_product_id found for recommended vendor_product_id {rec_vendor_product_id}. Skipping.")
                continue

            rec_master_product_id = vendor_to_master_map[rec_vendor_product_id]

            insert_data.append((
                master_product_id,  # target_id - using master_product_id
                'product',  # target_type
                rec_master_product_id,  # recommended_product_id - using master_product_id
                'similar',  # recommendation_type
                None,  # user_id (null for non-personalized recommendations)
                rank,  # rank
                score,  # score
                True,  # status
                current_time,  # created_at
                current_time  # updated_at
            ))

    # Batch insert using executemany
    pg_cur.executemany("""
        INSERT INTO product_recommendations
        (target_id, target_type, recommended_product_id, recommendation_type, 
         user_id, rank, score, status, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, insert_data)

    # Commit the changes
    pg_conn.commit()
    pg_cur.close()
    pg_conn.close()

    print(f"Successfully stored {len(insert_data)} recommendations in the database.")


def generate_all_recommendations(similarity_matrix, product_map, vendor_to_master_map, product_df=None, top_n=5, min_score=0.01):
    """
    Generate recommendations for all products in the product map.

    Args:
        similarity_matrix: The precomputed similarity matrix
        product_map: Mapping from product_id to matrix index
        vendor_to_master_map: Mapping from vendor_product_id to master_product_id
        product_df: DataFrame with product data (for fallback strategies)
        top_n: Maximum number of recommendations per product
        min_score: Minimum similarity score threshold

    Returns:
        A dictionary with product_id as key and a list of
        (recommended_product_id, score) tuples as value.
    """
    all_recommendations = {}
    total_products = len(product_map)

    print(f"Generating recommendations for {total_products} products...")

    # Track statistics
    products_with_recommendations = 0
    products_without_recommendations = 0
    total_recommendations = 0
    zero_score_products = []

    for i, (product_id, _) in enumerate(product_map.items()):
        if i % 20 == 0 or i == total_products - 1:
            print(f"Progress: {i+1}/{total_products} products processed")

        recommendations = recommend_products(product_id, similarity_matrix, product_map, top_n, min_score)

        if recommendations:
            all_recommendations[product_id] = recommendations
            products_with_recommendations += 1
            total_recommendations += len(recommendations)
        else:
            products_without_recommendations += 1
            zero_score_products.append(product_id)

            # Implement fallback strategy for products with no recommendations
            if product_df is not None:
                # Fallback to category-based or popularity-based recommendations
                # This is where we could implement a fallback strategy
                pass

    # Calculate average recommendations per product
    avg_recs = total_recommendations / products_with_recommendations if products_with_recommendations > 0 else 0

    print(f"\nRecommendation Generation Summary:")
    print(f"- Products with recommendations: {products_with_recommendations} ({products_with_recommendations/total_products*100:.1f}%)")
    print(f"- Products without recommendations: {products_without_recommendations} ({products_without_recommendations/total_products*100:.1f}%)")
    print(f"- Total recommendations generated: {total_recommendations}")
    print(f"- Average recommendations per product: {avg_recs:.2f}")

    if len(zero_score_products) > 0 and len(zero_score_products) <= 10:
        print(f"- Products with no recommendations: {zero_score_products}")
    elif len(zero_score_products) > 10:
        print(f"- First 10 products with no recommendations: {zero_score_products[:10]}...")

    return all_recommendations


if __name__ == "__main__":
    print("Starting the recommendation system...")
    interactions_df, product_df = generate_data()
    vendor_to_master_map = interactions_df[['vendor_product_id', 'master_product_id']].drop_duplicates().set_index('vendor_product_id')['master_product_id'].to_dict()

    print(f"Data loaded: {len(interactions_df)} interactions, {len(vendor_to_master_map)} unique vendor products")
    print(f"Vendor-to-Master product mapping created for {len(vendor_to_master_map)} products")

    # Analyze the interactions data
    print("\nAnalyzing interaction data:")
    interactions_per_product = interactions_df.groupby('vendor_product_id').size()
    interactions_per_user = interactions_df.groupby('user_id').size()
    print(f"- Number of unique vendor products with interactions: {len(interactions_per_product)}")
    print(f"- Number of unique users with interactions: {len(interactions_per_user)}")
    print(f"- Average interactions per vendor product: {interactions_per_product.mean():.2f}")
    print(f"- Average interactions per user: {interactions_per_user.mean():.2f}")
    print(f"- Most interactions for a vendor product: {interactions_per_product.max()}")
    print(f"- Most interactions for a user: {interactions_per_user.max()}")

    # Evaluate and save models
    similarity_matrix, product_map = evaluate_models(interactions_df, product_df)
    print(f"Models evaluated: {len(product_map)} products in the similarity matrix")

    # Analyze the similarity matrix
    print("\nAnalyzing similarity matrix:")
    non_zero_similarities = np.count_nonzero(similarity_matrix - np.eye(similarity_matrix.shape[0]))
    total_similarities = similarity_matrix.size - similarity_matrix.shape[0]  # exclude diagonal
    if total_similarities > 0:
        non_zero_percentage = (non_zero_similarities / total_similarities) * 100
        print(f"- Non-zero similarities: {non_zero_similarities} out of {total_similarities} ({non_zero_percentage:.2f}%)")
        print(f"- Mean similarity value (excluding diagonal): {np.mean(similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]):.4f}")
        print(f"- Max similarity value (excluding diagonal): {np.max(similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]):.4f}")

    # Generate recommendations with a small minimum score threshold
    min_similarity_score = 0.01  # This is a small threshold to filter out exact zeros
    print("\nGenerating recommendations for all products...")
    all_recommendations = generate_all_recommendations(
        similarity_matrix,
        product_map,
        vendor_to_master_map,
        product_df=product_df,
        top_n=5,
        min_score=min_similarity_score
    )

    # Store all recommendations in the database
    if all_recommendations:
        print("\nStoring recommendations in the database...")
        store_recommendations_in_db(all_recommendations, vendor_to_master_map)
        print("Recommendation system completed successfully.")
    else:
        print("\nNo recommendations were generated. Check the data and similarity calculations.")

    # Interactive testing section (optional)
    while True:
        try:
            choice = input("\nDo you want to test a specific product ID? (y/n): ").lower()
            if choice != 'y':
                print("Exiting the recommendation system.")
                break

            product_id = int(input("Enter a product ID for recommendations: "))

            # For testing, use a very low threshold to see what's happening
            test_min_score = 0.0001
            recommendations = recommend_products(
                product_id,
                similarity_matrix,
                product_map,
                top_n=5,
                min_score=test_min_score
            )

            if recommendations:
                print(f"\nTop recommendations for Product ID {product_id}:")
                for i, (rec_id, score) in enumerate(recommendations, 1):
                    print(f"  {i}. Product ID: {rec_id}, Similarity Score: {score:.6f}")

                # For the first recommendation, show what makes it similar
                if len(recommendations) > 0:
                    rec_id, _ = recommendations[0]
                    if product_id in product_map and rec_id in product_map:
                        orig_idx = product_map[product_id]
                        rec_idx = product_map[rec_id]

                        print(f"\nAnalyzing why Product {rec_id} is recommended for Product {product_id}:")
                        print(f"Similarity score: {similarity_matrix[orig_idx][rec_idx]:.6f}")
            else:
                print(f"No recommendations found for Product ID {product_id} with similarity > {test_min_score}.")
                print(f"This could be because:")
                print(f"1. The product has no interactions")
                print(f"2. The product has interactions but no overlapping users with other products")
                print(f"3. The product is not in our similarity matrix")

                if product_id in product_map:
                    print(f"\nThe product is in our matrix at index {product_map[product_id]}")
                    orig_idx = product_map[product_id]
                    # Show the highest similarities for this product
                    sims = similarity_matrix[orig_idx]
                    top_indices = np.argsort(-sims)[1:6]  # Skip the first one (self)
                    print(f"Top 5 similarity scores: {sims[top_indices]}")
                else:
                    print(f"\nThe product is not in our similarity matrix.")
        except ValueError:
            print("Invalid input. Please enter a valid product ID.")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
