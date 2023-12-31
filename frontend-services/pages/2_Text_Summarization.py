import requests
import streamlit as st
import json

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("Text Summarization")

text=st.text_input("Enter the Text")
print(type(text))

if st.button("Summarize"):
    if text is not None:
        payload={
            'text': text,
            'min_length':30,
            'max_length':150
        }
        res= requests.post('http://text-summarization:81/input_string', json=payload, timeout=240)
        print(res.json())

        ans=st.write(f"Summarized text is ***{res.json()['data'][0]['text']}***")

