import streamlit as st
import numpy as np
import pandas as pd
from modeling import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Loading model with scaling
model = load_model('NoSystolic_ScaledModelSVC.pkl')

def predict(model, input_data):
    """
    Extracts the probability of having any sleep issue
    """
    prediction = model.predict_proba(input_data)
    return prediction[0][1]

def plot_filled_gender(male, percentage):

    fig, ax = plt.subplots(figsize=(1,2))
    fig.patch.set_alpha(0)

    img_path = 'male_silhouette.png' if (male==1) else 'female_silhouette.png'

    # Load image
    img = mpimg.imread(img_path)

    # Convert the entire image to gray while preserving the alpha (transparency) channel
    grayscale_img_with_alpha = np.zeros_like(img)
    grayscale_img_with_alpha[:, :, :3] = [0.05098039, 0.07058824, 0.50980392] # issue color
    grayscale_img_with_alpha[:, :, 3] = img[:, :, 3]

    # Calculate the y limit based on percentage
    ylim = int(img.shape[0] * (1 - (percentage / 100)))

    # Replace the grayscale values with the original values up to the y-limit
    grayscale_img_with_alpha[:ylim, :, :3] = [0,0,0]

    # Display the modified image
    ax.imshow(grayscale_img_with_alpha, interpolation='none')

    # Add the percentage text
    ax.text(0.5, 0.5, f'{percentage}%', ha='center', va='center', color='white', 
            fontweight='bold', fontsize=10, transform=ax.transAxes)

    # Remove axis
    ax.axis('off')
    
    return fig

# Set title
st.title("Sleep Issue Predictor")

with st.sidebar:
    # User inputs
    age = st.number_input("Enter your age", value=31, min_value=18, max_value=60)
    sleep_duration = st.number_input("Enter sleep duration", value=8.0)
    heart_rate = st.number_input("Enter heart rate", value=70, min_value=60)
    daily_steps = st.number_input("Enter daily steps", value=8000)
    #systolic_bp = st.number_input("Enter systolic BP", value=120)
    is_male = st.selectbox("Select your gender", ["Male", "Female"])
    wf_technical = st.selectbox("Do you work in a technical field? such as: accounting, sofware, engineering, scientist...", ["Yes", "No"])

    # BMI Calculation
    # elevated_bmi = st.selectbox("Is your BMI elevated?", ["Yes", "No"])
    weight = st.number_input("What's your weight in Kg (kilograms)?", value=70.0)
    height = st.number_input("What's your height in M (meters)?", value=1.60)

BMI = weight/np.square(height)

elevated_bmi = 1 if BMI >= 25 else 0
# Bounds Reference: https://www.thecalculatorsite.com/articles/health/bmi-formula-for-bmi-calculations.php

# Convert categorical data to numeric
is_male = 1 if is_male == "Male" else 0
wf_technical = 1 if wf_technical == "Yes" else 0

# Predict button
if st.button("Predict"):

    # Getting input data
    input_data = np.array([[int(age), sleep_duration, heart_rate, int(daily_steps), is_male, elevated_bmi, wf_technical]])
    
    # Convert input_data to a DataFrame with appropriate column names
    columns = ['age', 'sleep_duration', 'heart_rate', 'daily_steps', 'is_male', 'elevated_bmi', 'wf_technical']
    input_df = pd.DataFrame(input_data, columns=columns)

    print(input_df)

    # Predicting ussing the model that already has scaling implemented in the pipeline
    prediction = predict(model, input_df)*100

    print(prediction)

    st.write(f"### Probability of having a sleep issue:")
    #st.write(f"## {prediction:.2f}%")

    # Creating figure
    fig = plot_filled_gender(is_male, np.round(prediction,2))

    # Creating column components for the plot
    col1, col2, col3 = st.columns(3)
    # Show plot in the middle of the app
    with col2:
        st.pyplot(fig, use_container_width=False)
    
