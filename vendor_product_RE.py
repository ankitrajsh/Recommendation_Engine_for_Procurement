    import pandas as pd
    import psycopg2
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm

    class FastVendorProductRecommender:
        def __init__(self):
            self.db_params = {
                'database': 'analytics_database',
                'user': 'postgres',
                'password': 'Y5mnshpDFF44',
                'host': '34.93.93.62',
                'port': '5432'
            }
            self.vendor_product_matrix = None
            self.product_df = None
            self.threshold = 0.8
            self.co_occurrence_dict = None
            self.product_to_vendors = {}
            self.vendor_to_products = {}

        def fetch_data(self):
            """Fetch vendor and product data from PostgreSQL using optimized query"""
            conn = psycopg2.connect(**self.db_params)
            try:
                query = """
                    SELECT vendor_name, product_name
                    FROM vendor_products_catalog
                    WHERE vendor_name IS NOT NULL 
                    AND product_name IS NOT NULL
                    GROUP BY vendor_name, product_name
                """
                print("Fetching data from database...")
                self.product_df = pd.read_sql(query, conn)
                print(f"Fetched {len(self.product_df)} unique vendor-product pairs")
            finally:
                conn.close()

            # Create efficient lookup dictionaries
            print("Building lookup dictionaries...")
            for _, row in self.product_df.iterrows():
                vendor, product = row['vendor_name'], row['product_name']

                # Product to vendors mapping
                if product not in self.product_to_vendors:
                    self.product_to_vendors[product] = set()
                self.product_to_vendors[product].add(vendor)

                # Vendor to products mapping
                if vendor not in self.vendor_to_products:
                    self.vendor_to_products[vendor] = set()
                self.vendor_to_products[vendor].add(product)

            return self.product_df

        def calculate_co_occurrences(self):
            """Calculate co-occurrences using optimized set operations"""
            print("Calculating product co-occurrences...")
            self.co_occurrence_dict = defaultdict(dict)

            # Get products with enough vendors
            valid_products = [
                product for product, vendors in self.product_to_vendors.items()
                if len(vendors) >= 2
            ]

            # Calculate co-occurrences using set operations
            for product1 in tqdm(valid_products):
                vendors_with_product1 = self.product_to_vendors[product1]
                vendors_count1 = len(vendors_with_product1)

                for product2 in valid_products:
                    if product1 != product2:
                        vendors_with_product2 = self.product_to_vendors[product2]
                        # Fast set intersection
                        vendors_with_both = len(vendors_with_product1 & vendors_with_product2)

                        if vendors_with_both > 0:
                            percentage = vendors_with_both / vendors_count1
                            if percentage >= self.threshold:
                                self.co_occurrence_dict[product1][product2] = percentage

        def get_recommendations(self, vendor_name, product_name):
            """Get recommendations using dictionary lookups"""
            try:
                # Initialize co-occurrences if not already done
                if self.co_occurrence_dict is None:
                    self.calculate_co_occurrences()

                # Get current vendor's products using fast lookup
                current_products = self.vendor_to_products.get(vendor_name, set())

                recommendations = []
                if product_name in self.co_occurrence_dict:
                    for related_product, percentage in self.co_occurrence_dict[product_name].items():
                        if related_product not in current_products:
                            recommendations.append({
                                'product': related_product,
                                'confidence': percentage,
                                'message': f"{percentage*100:.1f}% of vendors who have '{product_name}' "
                                         f"also have added '{related_product}'"
                            })

                return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)

            except Exception as e:
                print(f"Error getting recommendations: {e}")
                return []

    def main():
        recommender = FastVendorProductRecommender()

        try:
            # One-time initialization
            print("\nInitializing recommender system...")
            recommender.fetch_data()
            recommender.calculate_co_occurrences()
            print("\nInitialization complete! Ready for recommendations.\n")

            while True:
                vendor_name = input("Enter vendor name (or 'quit' to exit): ").strip()
                if vendor_name.lower() == 'quit':
                    break

                product_name = input("Enter product name: ").strip()

                print("\nFinding recommendations...")
                recommendations = recommender.get_recommendations(vendor_name, product_name)

                if recommendations:
                    print("\nRecommendations:")
                    for rec in recommendations:
                        print(f"{rec['message']}")
                else:
                    print("\nNo recommendations found for this product.")
                print("\n" + "-"*50)

        except Exception as e:
            print(f"An error occurred: {e}")

    if __name__ == "__main__":
        main()
