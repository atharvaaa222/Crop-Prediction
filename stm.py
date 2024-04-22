import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the trained model
def load_model(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

# Define the prediction function
def predict_crop(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Define the Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>Crop Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter the details below to predict the crop</p>", unsafe_allow_html=True)

    # Load the dataset
    df = pd.read_csv("Crop.csv")

    # Split the dataset into features and target
    X = df.drop(columns=['label'])
    y = df['label']

    # Load the trained model
    clf = load_model(X, y)

    # Custom layout for input fields
    # st.text("Pass Your Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen ratio")
        P = st.number_input("Phosphorous ratio")
        K = st.number_input("Potassium ratio")
    with col2:
        temperature = st.number_input("Temperature (°C)")
        humidity = st.number_input("Humidity (%)")
        rainfall = st.number_input("Rainfall (mm)")
    with col3:
        ph = st.number_input("pH value")

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Make prediction
    button_col = st.columns(3)
    with button_col[1]:
        if st.button("Forecast"):
            prediction = predict_crop(clf, input_data)
            st.success(f"Sow the seeds of success with the perfect timing for {prediction} crop cultivation")
            st.markdown(":yellow[Growing Smarter Farming Better]")


if __name__ == "__main__":
    main()
