import streamlit as st
import numpy as np
import pandas as pd
from modeling import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Loading issue model with scaling
issue_model = load_model('models/NoSystolic_ScaledModelSVC.pkl')

# Loading type of issue model with scaling
issue_type_model = load_model('models/SleepIssueType_ModelScaled.pkl')

def proba_predict(model, input_data):
    """
    Extracts the probability of having any sleep issue
    """
    prediction = model.predict_proba(input_data)
    return prediction[0][1]

def plot_filled_gender(male, percentage):

    fig, ax = plt.subplots(figsize=(1,2))
    fig.patch.set_alpha(0)

    img_path = 'images/male_silhouette.png' if (male==1) else 'images/female_silhouette.png'

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

def sleep_issue_image(issue, TITLE):

    fig, ax = plt.subplots(figsize=(2,2))
    fig.patch.set_alpha(0)

    img_path = 'images/sleep_apnea.jpeg' if (issue=='Sleep Apnea') else 'images/insomnia.jpeg'

    # Load image
    img = mpimg.imread(img_path)

    # Display the modified image
    ax.imshow(img, interpolation='none')

    # Remove axis
    ax.axis('off')

    # Comment on the issue for the user
    ax.set_title(TITLE, fontsize=8, color="#FFFFFF")
    
    return fig

# Set title
st.title("Sleep Issue Predictor")

with st.sidebar:
    # User inputs
    age = st.number_input("Enter your age", value=31, min_value=18, max_value=60)
    sleep_duration = st.number_input("Enter sleep duration", value=8.0)
    heart_rate = st.number_input("Enter heart rate", value=70, min_value=60)
    daily_steps = st.number_input("Enter daily steps", value=8000)
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
    issue_prob = proba_predict(issue_model, input_df)*100

    print(issue_prob)

    st.write(f"### Probability of having a sleep issue:")

    # Creating figure
    fig = plot_filled_gender(is_male, np.round(issue_prob,2))

    # This value will control the layout and will give the user extra information 
    # on the sleep condition it suffers ussing the logistic model.
    cut_off = 50

    # Creating column components for the plot
    col1, col2, col3 = st.columns([.2,.8,.1])
    # Show plot in the middle of the app
    with col2:
        st.pyplot(fig, use_container_width=False)

        #sleep_apnea=0 vs insomnia=1
        issue_prediction = issue_type_model.predict(input_df)[0]
        print(issue_prediction)
        print(issue_type_model.predict_proba(input_df))

        issue_prediction = "Sleep Apnea" if (issue_prediction == 1) else "Insomnia"

        if (issue_prob >= cut_off):
            #st.write(f"It is likely that your sleep issue is {issue_prediction}")

            title = f"You may have {issue_prediction}"

            st.pyplot( sleep_issue_image(issue_prediction, title), use_container_width=False)
        
        # Bringing image
        #fig, ax = plt.subplots(figsize=(1,2))
        #ax.imshow(grayscale_img_with_alpha, interpolation='none')
            