import streamlit as st
import pandas as pd
from backend import generate_response

st.write("Here's our first attempt to use LLM to simplyfy our task:")
user_input = st.text_area('Ask me Something')
if st.button('Generate'):
    result = generate_response(user_input)
    st.write(result)
else:
    st.write("Please provide input")