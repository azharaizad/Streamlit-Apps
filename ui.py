import streamlit as st
import pandas as pd
import pickle

# Load the dataset
laptops_df = pd.read_csv('laptops.csv')

# Load the trained model and label encoders
model = pickle.load('price_prediction_model.pkl')
label_encoders = pickle.load('label_encoders.pkl')

st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio("Go to", ["Home", "Laptop Suggestion"])

# Home Tab
if selected_tab == "Home":
    st.title("Welcome to Laptop Suggestion Based on SEMMA")
    st.write("Your one-stop solution to find the best laptop based on your preferences.")

    with st.expander("Project Description"):
        st.write("""
                 This project leverages the SEMMA methodology to help users select the best laptops based on their preferences."
                 The SEMMA (Sample, Explore, Modify, Model, Assess) methodology offers a structured approach to solving data 
                 analytics problems, enabling the effective application of machine learning techniques. This project applies 
                 SEMMA to predict laptop prices using two advanced machine learning algorithms, Random Gradient Boosting 
                 Regressor (RGBRegressor) and LightGBM (LGBM), and evaluates the accuracy of predictions compared to actual 
                 prices. Accurate price predictions are crucial for e-commerce platforms and consumers alike, helping to 
                 optimize pricing strategies and ensure competitive market positioning.

                 \nThe project begins by sampling a publicly available dataset containing laptop specifications and 
                 corresponding prices to create a manageable and representative dataset. The exploration phase 
                 involves analyzing key features, such as processor type, RAM, storage capacity, and brand, to 
                 understand their relationship with pricing. In the modification phase, the data undergoes 
                 preprocessing, including handling missing values, encoding categorical variables, and scaling 
                 numerical features. Using this cleaned dataset, RGBRegressor and LGBM models are trained to 
                 predict laptop prices. Finally, the models are assessed based on their accuracy and performance 
                 metrics, such as RMSE and R², with a comparative analysis highlighting their effectiveness in 
                 predicting prices. This process not only demonstrates the practical application of SEMMA but also 
                 provides valuable insights for businesses and consumers in the tech market.
                """)
        st.image("desc.png")

    with st.expander("User Guidelines"):
        st.write("- Use the filters in the Laptop Suggestion tab to customize your preferences.\n"
                 "- You can skip any filter if you are unsure.\n"
                 "- View the results in real-time based on your selections.")

# Laptop Suggestion Tab
elif selected_tab == "Laptop Suggestion":
    st.title("Laptop Suggestion")
    st.write("Select your preferences below to get suggestions.")

    # Filters
    st.subheader("Filters")

    # Brand filter
    brands = laptops_df['brand'].unique().tolist()
    selected_brands = st.multiselect("Select Brand(s)", options=brands, default=[])

    # Filter the dataset by selected brands for dynamic options
    filtered_df = laptops_df[laptops_df['brand'].isin(selected_brands)] if selected_brands else laptops_df

    # Price range filter
    if not filtered_df.empty:
        min_price, max_price = int(filtered_df['Price'].min()), int(filtered_df['Price'].max())
        selected_price = st.slider("Select Price Range (Rupee)", min_price, max_price, (min_price, max_price))
        filtered_df = filtered_df[filtered_df['Price'].between(selected_price[0], selected_price[1])]

    # Processor filter (dynamic based on previous filters)
    if not filtered_df.empty:
        processors = filtered_df['processor_brand'].unique().tolist()
        selected_processors = st.multiselect("Select Processor(s)", options=processors, default=[])
        filtered_df = filtered_df[filtered_df['processor_brand'].isin(selected_processors)] if selected_processors else filtered_df

    # Processor tier filter (dynamic based on previous filters)
    if not filtered_df.empty:
        processor_tiers = filtered_df['processor_tier'].unique().tolist()
        selected_processor_tiers = st.multiselect("Select Processor Tier(s)", options=processor_tiers, default=[])
        filtered_df = filtered_df[filtered_df['processor_tier'].isin(selected_processor_tiers)] if selected_processor_tiers else filtered_df

    # RAM memory filter (dynamic based on previous filters)
    if not filtered_df.empty:
        ram_options = filtered_df['ram_memory'].unique().tolist()
        selected_ram = st.multiselect("Select RAM Memory", options=ram_options, default=[])
        filtered_df = filtered_df[filtered_df['ram_memory'].isin(selected_ram)] if selected_ram else filtered_df

    # Primary storage filter (dynamic based on previous filters)
    if not filtered_df.empty:
        storage_options = filtered_df['primary_storage_capacity'].unique().tolist()
        selected_storage = st.multiselect("Select Primary Storage", options=storage_options, default=[])
        filtered_df = filtered_df[filtered_df['primary_storage_capacity'].isin(selected_storage)] if selected_storage else filtered_df

    # GPU brand filter (dynamic based on previous filters)
    if not filtered_df.empty:
        gpu_brands = filtered_df['gpu_brand'].unique().tolist()
        selected_gpu_brands = st.multiselect("Select GPU Brand(s)", options=gpu_brands, default=[])
        filtered_df = filtered_df[filtered_df['gpu_brand'].isin(selected_gpu_brands)] if selected_gpu_brands else filtered_df

    # Display filtered models
    st.subheader("Available Models")
    if not filtered_df.empty:
        st.write(filtered_df)

        # Prepare features for prediction
        features = filtered_df.drop(columns=['Price'], errors='ignore').copy()

        # Apply label encoding to categorical columns
        for column in label_encoders:
            if column in features.columns:
                try:
                    features[column] = label_encoders[column].transform(features[column])
                except ValueError:
                    st.warning(f"Unrecognized value in {column}. Ignoring rows with invalid values.")
                    features = features[features[column].isin(label_encoders[column].classes_)]

        # Predict prices
        filtered_df['Predicted Price'] = model.predict(features)

        # Display the table with predicted prices
        st.subheader("Predicted Price against Actual Price")
        st.write(filtered_df[['Predicted Price', 'Price']])

    else:
        st.write("No laptops match your criteria. Please adjust your filters.")
