import requests
import streamlit as st
import json

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("Texy Summarization")

text=st.text_input("Enter the Text")
print(type(text))

if st.button("Summarize"):
    if text is not None:
        payload={
            'text': text,
            'min_length':30,
            'max_length':150
        }
        res= requests.post('http://127.0.0.1:81/input_string', json=payload, timeout=60)
        print(res.json())

        ans=st.write(f"Summarized text is ***{res.json()['data'][0]['text']}***")

