import streamlit as st
import pickle

# --- Caching the Model ---
#Cache the model because it's large and slow to load.
@st.cache_resource
def load_model():
    """
    Loads the pickled machine learning model from the file.
    """
    try:
        with open('emotion_classifier.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'emotion_classifier.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None




model = load_model()


emotion_map = {
    0: 'sadness',
    1: 'anger',
    2: 'love',
    3: 'surprise',
    4: 'fear',
    5: 'joy'
}


# --- Streamlit User Interface ---

# Set the title of the web app
st.title("Emotion Detector from Text")

# Add a brief description of the app
st.write(
    "This application uses a Machine Learning model (Support Vector Machine) "
    "to predict the emotion conveyed in a piece of text. "
    "Enter a sentence below and click 'Predict Emotion' to see the result."
)

# Create a text area for the user to input their text
user_text = st.text_area("Enter your text here:", "I am overjoyed to see you again!")

# Create a button that the user will click to get the prediction
if st.button("Predict Emotion"):
    # Check if the model was loaded successfully and if the user entered text
    if model is not None and user_text:
        try:
            # The model's pipeline expects a list of texts for prediction.
            prediction_numerical = model.predict([user_text])[0]
            
            # Use the hardcoded emotion map to get the corresponding emotion name.
            prediction_emotional = emotion_map.get(prediction_numerical, "Unknown Emotion")
            
            # Display the final prediction in a styled success box.
            st.success(f"Predicted Emotion: **{prediction_emotional.capitalize()}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    elif not user_text:
        # If the user clicks the button without entering text, show a warning.
        st.warning("Please enter some text before predicting.")

