
# Updated 20240627 Shanghai Time 08:00
# adding streamlit interface to browse web documents
# add 'search_query' : q, in the metadata
# ⚠️ Segregating display datastructure from saved db 🗃️ datastructure
# REV02 - fix old type of DDG
import streamlit as st
import os
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
from rich.console import Console
from newspaper import Article
import pickle
from langchain.schema.document import Document
console = Console(width=90)
from io import StringIO
from io import BytesIO
import random
import string
import datetime
import base64


st.set_page_config(page_title="AI powered web serach", layout="wide",page_icon='📱')

if "keyDDGfile" not in st.session_state:
    st.session_state.keyDDGfile = 1  


st.title("AI powered Web Document Search")
st.write('Using Newspaper3k and DuckDuckGo LangChain wrapper')
st.divider()
# Upload the audio file
file1 = st.sidebar.file_uploader("Upload a text document", 
                                    type=["ddg"],accept_multiple_files=False, 
                                    key=st.session_state.keyDDGfile)
st.divider()
if file1:
    console.print(f'[blink2 orange1]Loading...')
    pkl_file = open(file1.name, 'rb')
    st.write(type(pkl_file))
    data_docs = pickle.load(pkl_file)
    st.write(type(data_docs))
    pkl_file.close()
    st.write(data_docs)