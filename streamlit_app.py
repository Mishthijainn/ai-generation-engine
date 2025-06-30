
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from personalization_engine import PersonalizationEngine

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Personalization Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the models and data


@st.cache_resource
def load_models():
    try:
        users_df = pd.read_csv('users_data.csv')
        products_df = pd.read_csv('products_data.csv')
        interactions_df = pd.read_csv('interactions_data.csv')

        # user_item_matrix = pd.read_pickle('user_item_matrix.pkl')
        user_similarity = joblib.load('user_similarity_matrix.pkl')
        item_similarity = joblib.load('item_similarity_matrix.pkl')
        product_similarity = joblib.load('product_similarity_matrix.pkl')

        le_gender = joblib.load('gender_encoder.pkl')
        le_income = joblib.load('income_encoder.pkl')
        le_device = joblib.load('device_encoder.pkl')
        scaler = joblib.load('user_scaler.pkl')
        kmeans = joblib.load('collaborative_filtering_model.pkl')
        segment_names = joblib.load('user_segmentation_model.pkl')
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )

        rec_engine = PersonalizationEngine(
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
        )

        return rec_engine, users_df, products_df, interactions_df

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


# Main app


def main():
    st.title("🤖 AI-Powered Hyper-Personalized Landing Page Generator")
    st.markdown("### Advanced ML/DL Models for E-commerce Personalization")

    # Load models
    rec_engine, users_df, products_df, interactions_df = load_models()

    if rec_engine is None:
        st.error("Unable to load models. Please ensure all model files are present.")
        return

    # Sidebar for user input
    st.sidebar.header("🔧 Configuration")

    # User type selection
    user_type = st.sidebar.radio(
        "Select User Type:",
        ["New User (Cold Start)", "Existing User"]
    )

    recommendations = None

    if user_type == "New User (Cold Start)":
        st.sidebar.subheader("📋 User Demographics")

        # Demographic inputs
        age = st.sidebar.slider("Age", 18, 65, 30)
        gender = st.sidebar.selectbox("Gender", ["M", "F"])
        income_group = st.sidebar.selectbox(
            "Income Group", ["Low", "Medium", "High"])
        device_type = st.sidebar.selectbox(
            "Device Type", ["Mobile", "Desktop"])

        if st.sidebar.button("🎯 Get Personalized Recommendations", type="primary"):
            user_demographics = {
                'age': age,
                'gender': gender,
                'income_group': income_group,
                'device_type': device_type
            }

            with st.spinner("🧠 AI is analyzing your profile..."):
                recommendations = rec_engine.handle_cold_start(
                    user_demographics)

    else:  # Existing User
        st.sidebar.subheader("👤 Select Existing User")

        if users_df is not None:
            selected_user = st.sidebar.selectbox(
                "User ID",
                users_df['user_id'].tolist()
            )

            # Show user details
            user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
            st.sidebar.write(f"**Age:** {user_info['age']}")
            st.sidebar.write(f"**Gender:** {user_info['gender']}")
            st.sidebar.write(f"**Income:** {user_info['income_group']}")
            st.sidebar.write(f"**Segment:** {user_info['segment_name']}")

            if st.sidebar.button("🎯 Get Recommendations", type="primary"):
                with st.spinner("🔍 Finding personalized recommendations..."):
                    recommendations = rec_engine.get_user_recommendations(
                        selected_user)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        if recommendations:
            st.header("🎯 Personalized Recommendations")

            # Display JSON output (as requested)
            st.subheader("📄 JSON Output")
            st.json(recommendations)

            # Display recommendations in a nice format
            st.subheader("🛍️ Recommended Products")

            if recommendations['recommendations']:
                for i, product in enumerate(recommendations['recommendations'], 1):
                    with st.expander(f"#{i} {product['category']} - {product['brand']}", expanded=True):
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.write(
                                f"**Product ID:** {product['product_id']}")
                            st.write(f"**Category:** {product['category']}")
                            st.write(f"**Brand:** {product['brand']}")
                        with col_b:
                            st.metric("Price", f"${product['price']:.2f}")
                        with col_c:
                            st.metric("Rating", f"{product['rating']}/5.0")

            # Algorithm insights
            st.subheader("🧠 AI Algorithm Insights")
            col_alg1, col_alg2, col_alg3 = st.columns(3)

            with col_alg1:
                st.metric("Confidence Score",
                          f"{recommendations['confidence_score']:.1%}")

            with col_alg2:
                algorithms_used = ", ".join(recommendations['algorithm_used'])
                st.write(f"**Algorithms Used:**")
                st.write(algorithms_used)

            with col_alg3:
                if 'predicted_segment' in recommendations:
                    st.write(f"**Predicted Segment:**")
                    st.write(recommendations['predicted_segment'])

        else:
            st.info(
                "👈 Please configure user details in the sidebar to get personalized recommendations.")

            # Show model capabilities
            st.header("🧠 AI Model Capabilities")

            capabilities = [
                "**User Segmentation:** K-Means clustering with 5 distinct user segments",
                "**Collaborative Filtering:** Matrix factorization using Non-negative Matrix Factorization",
                "**Content-Based Filtering:** Product similarity using cosine similarity",
                "**Cold Start Handling:** Demographic-based recommendations for new users",
                "**Hybrid Approach:** Combines multiple algorithms for robust recommendations",
                "**Real-time Processing:** Sub-second response times for personalization"
            ]

            for capability in capabilities:
                st.markdown(f"✅ {capability}")

    with col2:
        st.header("📊 Model Statistics")

        if users_df is not None and products_df is not None:
            # Dataset stats
            st.metric("Total Users", len(users_df))
            st.metric("Total Products", len(products_df))
            st.metric("Total Interactions", len(interactions_df))

            # User segments distribution
            st.subheader("👥 User Segments")
            segment_counts = users_df['segment_name'].value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="User Segment Distribution"
            )
            st.plotly_chart(fig_segments, use_container_width=True)

            # Product categories
            st.subheader("🛍️ Product Categories")
            category_counts = products_df['category'].value_counts()
            fig_categories = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Products by Category"
            )
            fig_categories.update_layout(
                xaxis_title="Category",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_categories, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔧 Technical Details

    This AI-powered personalization engine uses advanced machine learning algorithms to provide 
    hyper-personalized product recommendations. The system handles both existing users through 
    collaborative filtering and new users through sophisticated cold-start strategies.

    **Key Features:**
    - Real ML/DL models
    - Sub-second response times
    - JSON output format
    - Advanced cold start problem solving
    - Enterprise-grade recommendation accuracy
    """)


if __name__ == "__main__":
    main()
