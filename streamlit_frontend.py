import streamlit as st
import requests
import json

st.set_page_config(page_title="Linear Regression Predictor", layout="centered")

st.title("📊 Linear Regression Predictor")
st.write("Enter the 4 feature values to get a prediction")

# API endpoint
api_url = "http://127.0.0.1:8000/predict"

# Input fields
col1, col2 = st.columns(2)
with col1:
    f1 = st.number_input("Feature 1", value=45.67, step=0.01)
    f2 = st.number_input("Feature 2", value=23.45, step=0.01)

with col2:
    f3 = st.number_input("Feature 3", value=78.90, step=0.01)
    f4 = st.number_input("Feature 4", value=12.34, step=0.01)

# Predict button
if st.button("🔮 Predict", type="primary"):
    try:
        # Prepare data
        data = {
            "features": [[f1, f2, f3, f4]]
        }
        
        # Make API request
        with st.spinner("Making prediction..."):
            response = requests.post(api_url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            
            # Display result
            st.success("✅ Prediction successful!")
            st.metric(label="Prediction", value=f"${prediction:,.2f}")
            
            # Show details
            with st.expander("📝 Details"):
                st.json({
                    "Input": [f1, f2, f3, f4],
                    "Prediction": prediction
                })
        else:
            st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Sample data button
if st.button("📥 Load Sample Data"):
    st.session_state.f1 = 45.67
    st.session_state.f2 = 23.45
    st.session_state.f3 = 78.90
    st.session_state.f4 = 12.34
    st.rerun()

st.divider()
st.caption("Make sure the FastAPI server is running on http://127.0.0.1:8000")