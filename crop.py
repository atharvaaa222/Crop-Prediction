import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("Crop.csv")

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

def predict_crop():
    # Take input for each feature
    N = float(input("Enter ratio of Nitrogen content in soil: "))
    P = float(input("Enter ratio of Phosphorous content in soil: "))
    K = float(input("Enter ratio of Potassium content in soil: "))
    temperature = float(input("Enter temperature in degree Celsius: "))
    humidity = float(input("Enter relative humidity in %: "))
    ph = float(input("Enter ph value of the soil: "))
    rainfall = float(input("Enter rainfall in mm: "))

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
    prediction = clf.predict(input_data)

    return prediction[0]  # Return the predicted crop name

# Example usage:
predicted_crop = predict_crop()
print("Predicted crop:", predicted_crop)
