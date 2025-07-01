# Collaborative Filtering Model
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity  # ✅ Correct

import numpy as np
users_df = pd.read_csv('users_data.csv')
products_df = pd.read_csv('products_data.csv')
interactions_df = pd.read_csv('interactions_data.csv')
print("Building Collaborative Filtering Model...")
interactions_df = pd.read_csv('interactions_data.csv')
# user_item_matrix = pd.read_pickle('user_item_matrix.pkl')
user_similarity = joblib.load('user_similarity_matrix.pkl')
item_similarity = joblib.load('item_similarity_matrix.pkl')
product_similarity = joblib.load('product_similarity_matrix.pkl')

le_gender = joblib.load('gender_encoder.pkl')
le_income = joblib.load('income_encoder.pkl')
le_device = joblib.load('device_encoder.pkl')
scaler = joblib.load('user_scaler.pkl')
# kmeans = joblib.load('collaborative_filtering_model.pkl')
kmeans = joblib.load('user_segmentation_model.pkl')

segment_names = joblib.load('user_segmentation_model.pkl')
# Create user-item matrix
user_item_matrix = interactions_df.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    fill_value=0
)

print(f"User-item matrix shape: {user_item_matrix.shape}")

# Use Non-negative Matrix Factorization for collaborative filtering
nmf_model = NMF(n_components=50, random_state=42, max_iter=200)

# Fit the model
user_features_cf = nmf_model.fit_transform(user_item_matrix)
item_features_cf = nmf_model.components_

print("Collaborative filtering model trained!")

# Calculate user similarity matrix
user_similarity = cosine_similarity(user_features_cf)
item_similarity = cosine_similarity(item_features_cf.T)

print(f"User similarity matrix shape: {user_similarity.shape}")
print(f"Item similarity matrix shape: {item_similarity.shape}")

# Save collaborative filtering model
joblib.dump(nmf_model, 'collaborative_filtering_model.pkl')
joblib.dump(user_similarity, 'user_similarity_matrix.pkl')
joblib.dump(item_similarity, 'item_similarity_matrix.pkl')

print("Collaborative filtering model saved!")

# 3. Content-Based Filtering Model
print("\n3. Building Content-Based Filtering Model...")

# Create product feature matrix

# Encode product categorical features
le_category = LabelEncoder()
le_brand = LabelEncoder()

product_features = pd.DataFrame({
    'category_encoded': le_category.fit_transform(products_df['category']),
    'brand_encoded': le_brand.fit_transform(products_df['brand']),
    'price_normalized': (products_df['price'] - products_df['price'].min()) / (products_df['price'].max() - products_df['price'].min()),
    'rating_normalized': (products_df['rating'] - products_df['rating'].min()) / (products_df['rating'].max() - products_df['rating'].min())
})

# Calculate product similarity based on features
product_similarity = cosine_similarity(product_features)

print(f"Product similarity matrix shape: {product_similarity.shape}")

# Save content-based filtering components
joblib.dump(le_category, 'category_encoder.pkl')
joblib.dump(le_brand, 'brand_encoder.pkl')
joblib.dump(product_similarity, 'product_similarity_matrix.pkl')

print("Content-based filtering model saved!")

# 4. Build recommendation functions
print("\n4. Building Recommendation Engine...")


class PersonalizationEngine:
    def __init__(
        self,
        users_df,
        products_df,
        interactions_df,
        user_item_matrix,
        user_similarity,
        item_similarity,
        product_similarity,
        le_gender,
        le_income,
        le_device,
        scaler,
        kmeans,
        segment_names
    ):
        self.users_df = users_df
        self.products_df = products_df
        self.interactions_df = interactions_df
        self.user_item_matrix = user_item_matrix
        self.user_similarity = user_similarity
        self.item_similarity = item_similarity
        self.product_similarity = product_similarity
        self.le_gender = le_gender
        self.le_income = le_income
        self.le_device = le_device
        self.scaler = scaler
        self.kmeans = kmeans
        self.segment_names = segment_names
        self.segment_preferences = self._build_segment_preferences()

    def _build_segment_preferences(self):
        segment_prefs = {}

        for segment in self.users_df['segment_name'].unique():
            segment_users = self.users_df[self.users_df['segment_name']
                                          == segment]['user_id'].values
            segment_interactions = self.interactions_df[self.interactions_df['user_id'].isin(
                segment_users)]
            popular_products = segment_interactions.groupby(
                'product_id')['rating'].agg(['mean', 'count']).reset_index()
            popular_products = popular_products[popular_products['count'] >= 3]
            popular_products = popular_products.sort_values(
                ['mean', 'count'], ascending=[False, False])

            segment_prefs[segment] = {
                'popular_products': popular_products['product_id'].head(10).tolist(),
                'avg_rating': popular_products['mean'].mean(),
                'total_interactions': len(segment_interactions)
            }

        return segment_prefs

    # def get_user_recommendations(self, user_id, n_recommendations=5):
    #     recommendations = {
    #         'user_id': user_id,
    #         'recommendations': [],
    #         'algorithm_used': [],
    #         'confidence_score': 0.0
    #     }

    #     if user_id in self.users_df['user_id'].values:
    #         user_idx = list(self.user_item_matrix.index).index(user_id)
    #         similar_users = np.argsort(
    #             self.user_similarity[user_idx])[::-1][1:6]

    #         recommended_products = []
    #         for similar_user_idx in similar_users:
    #             similar_user_id = self.user_item_matrix.index[similar_user_idx]
    #             user_products = self.interactions_df[self.interactions_df['user_id']
    #                                                  == similar_user_id]['product_id'].values
    #             recommended_products.extend(user_products)

    #         user_interacted_products = self.interactions_df[self.interactions_df['user_id']
    #                                                         == user_id]['product_id'].values
    #         recommended_products = list(
    #             set(recommended_products) - set(user_interacted_products))

    #         if len(recommended_products) >= n_recommendations:
    #             final_recommendations = recommended_products[:n_recommendations]
    #             recommendations['algorithm_used'].append(
    #                 'collaborative_filtering')
    #             recommendations['confidence_score'] = 0.8
    #         else:
    #             final_recommendations = recommended_products
    #             recommendations['algorithm_used'].append(
    #                 'collaborative_filtering_partial')
    #             recommendations['confidence_score'] = 0.6
    #     else:
    #         final_recommendations = []
    #         recommendations['confidence_score'] = 0.3

    #     if len(final_recommendations) < n_recommendations:
    #         user_info = self.users_df[self.users_df['user_id'] == user_id]
    #         if not user_info.empty:
    #             user_segment = user_info['segment_name'].iloc[0]
    #             segment_products = self.segment_preferences.get(
    #                 user_segment, {}).get('popular_products', [])
    #         else:
    #             segment_products = self.interactions_df.groupby('product_id')['rating'].mean(
    #             ).sort_values(ascending=False).head(10).index.tolist()

    #         remaining_slots = n_recommendations - len(final_recommendations)
    #         segment_recs = [
    #             p for p in segment_products if p not in final_recommendations][:remaining_slots]
    #         final_recommendations.extend(segment_recs)
    #         recommendations['algorithm_used'].append('demographic_fallback')

    #     for product_id in final_recommendations:
    #         product_info = self.products_df[self.products_df['product_id'] == product_id]
    #         if not product_info.empty:
    #             product_details = product_info.iloc[0]
    #             recommendations['recommendations'].append({
    #                 'product_id': product_id,
    #                 'category': product_details['category'],
    #                 'brand': product_details['brand'],
    #                 'price': product_details['price'],
    #                 'rating': product_details['rating']
    #             })

    #     return recommendations

    # def handle_cold_start(self, user_demographics):
    #     age = user_demographics.get('age', 30)
    #     gender = user_demographics.get('gender', 'M')
    #     income = user_demographics.get('income_group', 'Medium')
    #     device = user_demographics.get('device_type', 'Mobile')

    #     temp_user_features = np.array([[
    #         age,
    #         self.le_gender.transform(
    #             [gender])[0] if gender in self.le_gender.classes_ else 0,
    #         0,
    #         self.le_income.transform(
    #             [income])[0] if income in self.le_income.classes_ else 1,
    #         self.le_device.transform(
    #             [device])[0] if device in self.le_device.classes_ else 0
    #     ]])

    #     temp_user_scaled = self.scaler.transform(temp_user_features)
    #     predicted_segment = self.kmeans.predict(temp_user_scaled)[0]
    #     segment_name = self.segment_names[predicted_segment]
    #     segment_products = self.segment_preferences.get(
    #         segment_name, {}).get('popular_products', [])

    #     recommendations = {
    #         'user_id': 'new_user',
    #         'predicted_segment': segment_name,
    #         'recommendations': [],
    #         'algorithm_used': ['cold_start_demographic'],
    #         'confidence_score': 0.5
    #     }

    #     for product_id in segment_products[:5]:
    #         product_info = self.products_df[self.products_df['product_id'] == product_id]
    #         if not product_info.empty:
    #             product_details = product_info.iloc[0]
    #             recommendations['recommendations'].append({
    #                 'product_id': product_id,
    #                 'category': product_details['category'],
    #                 'brand': product_details['brand'],
    #                 'price': product_details['price'],
    #                 'rating': product_details['rating']
    #             })

    #     return recommendations

    # def __init__(self):
    #     self.users_df = users_df
    #     self.products_df = products_df
    #     self.interactions_df = interactions_df
    #     self.user_item_matrix = user_item_matrix
    #     self.user_similarity = user_similarity
    #     self.item_similarity = item_similarity
    #     self.product_similarity = product_similarity
    #     self.segment_preferences = self._build_segment_preferences()

    # def _build_segment_preferences(self):
    #     """Build preferences for each user segment"""
    #     segment_prefs = {}

    #     for segment in users_df['segment_name'].unique():
    #         segment_users = users_df[users_df['segment_name']
    #                                  == segment]['user_id'].values
    #         segment_interactions = interactions_df[interactions_df['user_id'].isin(
    #             segment_users)]

    #         # Get popular products for this segment
    #         popular_products = segment_interactions.groupby(
    #             'product_id')['rating'].agg(['mean', 'count']).reset_index()
    #         # Filter for reliability
    #         popular_products = popular_products[popular_products['count'] >= 3]
    #         popular_products = popular_products.sort_values(
    #             ['mean', 'count'], ascending=[False, False])

    #         segment_prefs[segment] = {
    #             'popular_products': popular_products['product_id'].head(10).tolist(),
    #             'avg_rating': popular_products['mean'].mean(),
    #             'total_interactions': len(segment_interactions)
    #         }

    #     return segment_prefs

    def get_user_recommendations(self, user_id, n_recommendations=5):
        """Get personalized recommendations for a user"""
        recommendations = {
            'user_id': user_id,
            'recommendations': [],
            'algorithm_used': [],
            'confidence_score': 0.0
        }

        # Check if user exists in our data
        if user_id in self.users_df['user_id'].values:
            # Existing user - use collaborative filtering
            user_idx = list(self.user_item_matrix.index).index(user_id)

            # Find similar users
            similar_users = np.argsort(self.user_similarity[user_idx])[
                ::-1][1:6]  # Top 5 similar users

            # Get recommendations from similar users
            recommended_products = []
            for similar_user_idx in similar_users:
                similar_user_id = self.user_item_matrix.index[similar_user_idx]
                user_products = self.interactions_df[self.interactions_df['user_id']
                                                     == similar_user_id]['product_id'].values
                recommended_products.extend(user_products)

            # Remove duplicates and products user has already interacted with
            user_interacted_products = self.interactions_df[self.interactions_df['user_id']
                                                            == user_id]['product_id'].values
            recommended_products = list(
                set(recommended_products) - set(user_interacted_products))

            if len(recommended_products) >= n_recommendations:
                final_recommendations = recommended_products[:n_recommendations]
                recommendations['algorithm_used'].append(
                    'collaborative_filtering')
                recommendations['confidence_score'] = 0.8
            else:
                final_recommendations = recommended_products
                recommendations['algorithm_used'].append(
                    'collaborative_filtering_partial')
                recommendations['confidence_score'] = 0.6
        else:
            final_recommendations = []
            recommendations['confidence_score'] = 0.3

        # Fill remaining recommendations with segment-based recommendations
        if len(final_recommendations) < n_recommendations:
            # Use demographic-based recommendations (cold start)
            user_info = self.users_df[self.users_df['user_id'] == user_id]
            if not user_info.empty:
                user_segment = user_info['segment_name'].iloc[0]
                segment_products = self.segment_preferences.get(
                    user_segment, {}).get('popular_products', [])
            else:
                # New user - use overall popular products
                segment_products = self.interactions_df.groupby('product_id')['rating'].mean(
                ).sort_values(ascending=False).head(10).index.tolist()

            # Add segment-based recommendations
            remaining_slots = n_recommendations - len(final_recommendations)
            segment_recs = [
                p for p in segment_products if p not in final_recommendations][:remaining_slots]
            final_recommendations.extend(segment_recs)
            recommendations['algorithm_used'].append('demographic_fallback')

        # Format final recommendations with product details
        for product_id in final_recommendations:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                product_details = product_info.iloc[0]
                recommendations['recommendations'].append({
                    'product_id': product_id,
                    'category': product_details['category'],
                    'brand': product_details['brand'],
                    'price': product_details['price'],
                    'rating': product_details['rating']
                })

        return recommendations

    def handle_cold_start(self, user_demographics):
        """Handle cold start problem for new users"""
        age = user_demographics.get('age', 30)
        gender = user_demographics.get('gender', 'M')
        income = user_demographics.get('income_group', 'Medium')
        device = user_demographics.get('device_type', 'Mobile')

        # Create temporary user profile
        temp_user_features = np.array([[
            age,
            le_gender.transform([gender])[
                0] if gender in le_gender.classes_ else 0,
            0,  # Default city
            le_income.transform([income])[
                0] if income in le_income.classes_ else 1,
            le_device.transform([device])[
                0] if device in le_device.classes_ else 0
        ]])

        # Scale features
        temp_user_scaled = scaler.transform(temp_user_features)

        # Predict segment
        predicted_segment = kmeans.predict(temp_user_scaled)[0]
        segment_name = segment_names[predicted_segment]

        # Get recommendations for this segment
        segment_products = self.segment_preferences.get(
            segment_name, {}).get('popular_products', [])

        recommendations = {
            'user_id': 'new_user',
            'predicted_segment': segment_name,
            'recommendations': [],
            'algorithm_used': ['cold_start_demographic'],
            'confidence_score': 0.5
        }

        # Format recommendations
        for product_id in segment_products[:5]:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                product_details = product_info.iloc[0]
                recommendations['recommendations'].append({
                    'product_id': product_id,
                    'category': product_details['category'],
                    'brand': product_details['brand'],
                    'price': product_details['price'],
                    'rating': product_details['rating']
                })

        return recommendations


# Initialize the recommendation engine
rec_engine = PersonalizationEngine(users_df,
                                   products_df,
                                   interactions_df,
                                   user_item_matrix,
                                   user_similarity,
                                   item_similarity,
                                   product_similarity,
                                   le_gender,
                                   le_income,
                                   le_device,
                                   scaler,
                                   kmeans,
                                   segment_names)

# Test the recommendation engine
print("Testing recommendation engine...")

# Test with existing user
test_user = users_df['user_id'].iloc[0]
recommendations = rec_engine.get_user_recommendations(test_user)
print(f"\nRecommendations for {test_user}:")
print(json.dumps(recommendations, indent=2))

print("\nRecommendation engine built successfully!")
