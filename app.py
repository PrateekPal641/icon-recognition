import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import inference
from langchain.chat_models import AzureChatOpenAI
import os
from langchain.schema import HumanMessage
import json


yolo_model = YOLO('best.pt')
roboflow_model = inference.get_model("web-icon-classification/1")
chat4 = AzureChatOpenAI(
    openai_api_base=os.environ['BASE_URL'],
    openai_api_version="2024-02-15-preview",
    deployment_name="gpt-4",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
    temperature=0,
    request_timeout=30,
    max_retries=3
) 

def initiate_prompt(icon_name):
    prompt =  '''Given the name of an app icon, return a list of alternative names that represent similar functionality in the context of a web or mobile app.

    User Input: "Settings"

    Expected Output: Generate a list of alternative names that convey the same or similar functionality as "Settings" in the context of web or mobile apps. 

    Model Response: {
    "alternatives": ["Preferences", "Options", "Controls", "Configuration", "Setup"]
    }

    User Input:''' + icon_name +'\n'+ "    Model Response:" 

    return prompt

st.title("App/Web Icon Classification Comparison")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify Image"):
        with st.spinner('Classifying...'):
            try:
                prediction = yolo_model(image)
                class_id_1 = prediction[0].names[prediction[0].probs.top1]
                classes_1 = json.loads(chat4.predict_messages(messages=[HumanMessage(content=initiate_prompt(class_id_1))]).content)['alternatives']
                classes_1.insert(0, class_id_1)
            except:
                classes_1 = "None"

            try:
                prediction = roboflow_model.infer(image)
                class_id_2 =  prediction[0].predicted_classes[0]
                classes_2 = json.loads(chat4.predict_messages(messages=[HumanMessage(content=initiate_prompt(class_id_2))]).content)['alternatives']
                classes_2.insert(0, class_id_2)
            except:
                classes_2 = "None"

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Yolov8-x Prediction")
                st.write(f"Predicted Class: {classes_1}")
            with col2:
                st.subheader("ViT Prediction")
                st.write(f"Predicted Class: {classes_2}")
