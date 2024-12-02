# src/app.py

import streamlit as st
from grammar_correction import correct_grammar, load_model

def main():
    # Load the model
    model, tokenizer = load_model()

    st.title("Context-Aware Grammar Correction System")

    # Text input
    input_text = st.text_area("Enter text to correct:", height=150)

    if st.button("Correct Grammar"):
        if input_text:
            corrected_text = correct_grammar(input_text, model, tokenizer)
            st.subheader("Corrected Text:")
            st.write(corrected_text)
        else:
            st.warning("Please enter some text to correct.")

    # Add information about the project
    st.sidebar.header("About this project")
    st.sidebar.markdown("""
    This Context-Aware Grammar Correction System:
    - Uses transformer-based architectures (T5) for real-time contextual language suggestions.
    - Is fine-tuned on custom datasets with diverse writing styles.
    - Provides an intuitive user interface for testing and feedback.
    - Ensures scalability and adaptability for personalized grammar assistance.
    """)

if __name__ == "__main__":
    main()