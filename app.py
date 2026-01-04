import streamlit as st
import torch
from transformers import pipeline
from transformers import AutoTokenizer

# Page Config
st.set_page_config(page_title="Suicide Severity Detector", page_icon="🎗️")

@st.cache_resource
def load_model():
    model_id = "tcxy98/suicide-severity-deberta"
    return pipeline("text-classification", model=model_id, device=-1)

classifier = load_model()

# Severity Label Mapping
severity_map = {
    "LABEL_0": "No suicidal risk",
    "LABEL_1": "Passive suicidal ideation",
    "LABEL_2": "Active ideation without plan",
    "LABEL_3": "Active ideation with method mentioned",
    "LABEL_4": "Active ideation with intent and plan",
    "LABEL_5": "Immediate crisis or final goodbye"
}

# UI Layout
st.title("🎗️ Reddit Severity Analysis")
st.write("Paste a post below to analyze the mental health severity level.")

user_input = st.text_area("Reddit Post Text:", height=300, placeholder="Paste content here...")

if st.button("Analyze Severity"):
    if user_input.strip():
        with st.spinner("Analyzing text..."):
            # Truncate input to 512 tokens (model limit)
            result = classifier(user_input[:2000])[0]
            
            label = result['label']
            score = result['score']
            description = severity_map.get(label, "Unknown")
            severity_level = int(label.split('_')[1]) # Extract 0-5

            # Visual Feedback
            st.divider()
            cols = st.columns([1, 4])
            
            with cols[0]:
                if severity_level >= 4:
                    st.error(f"LEVEL {severity_level}")
                elif severity_level >= 2:
                    st.warning(f"LEVEL {severity_level}")
                else:
                    st.success(f"LEVEL {severity_level}")
            
            with cols[1]:
                st.subheader(description)
                st.progress(severity_level / 5)
                st.caption(f"Confidence Score: {score:.2%}")
                
            if severity_level >= 4:
                st.info("**Action Recommended:** This post indicates a high crisis level. Consider immediate intervention protocols.")
    else:
        st.warning("Please enter some text first.")