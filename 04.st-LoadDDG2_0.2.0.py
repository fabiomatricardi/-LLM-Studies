
# Updated 20240628 Shanghai Time 12:00
# adding streamlit interface to browse web documents
# - tool to merge all .ddg file
# - tool to explore the web documents
# - select documents to build a temporary vectorstore
import streamlit as st
from rich.markdown import Markdown
from rich.console import Console
console = Console(width=90)
# DATA AND WARNING HANDLING
import warnings
warnings.filterwarnings(action='ignore')
import pickle
import os
import random
import string
import datetime
# LANGCHAIN IMPORTS
from langchain.schema.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_text_splitters import CharacterTextSplitter
# LLM AND NLP RELATED IMPORTS
from llama_cpp import Llama
from newspaper import Article
import tiktoken


# SET MAIN STREAMLIT CONFIGURATION
st.set_page_config(page_title="AI powered web serach", layout="wide",page_icon='📱')

# to write out a log file PLAIN TXT
def writehistory(filename,text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

# GENERATE RANDOM HASH of n charachters
def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

# DEFINE FILENAMES
if "sessionHASH" not in st.session_state:
    st.session_state.sessionHASH = genRANstring(5)
if "sessionlogfilename" not in st.session_state:
    st.session_state.sessionlogfilename = f'{st.session_state.sessionHASH}-log.txt'
if "sessionDBfilename" not in st.session_state:
    st.session_state.sessionDBfilename = f'{st.session_state.sessionHASH}_searchDB.ddg'

# Initialize search history
if "searches" not in st.session_state:
    st.session_state.searches = []

# Newsletter3k utility disctionary from NLTK
@st.cache_resource
def loadDictionary():
    import nltk
    nltk.download('punkt')
# Create Tokenizer conunter
@st.cache_resource
def tokencounter():
    import tiktoken
    #context_count = len(encoding.encode(yourtext))
    encoding = tiktoken.get_encoding("r50k_base") 
    return encoding

# Network Connector LangChain DuckDuckGo Wrapper
#from https://python.langchain.com/v0.1/docs/integrations/tools/ddg/
# https://pypi.org/project/duckduckgo-search
# https://api.python.langchain.com/en/latest/utilities/langchain_community.utilities.duckduckgo_search.DuckDuckGoSearchAPIWrapper.html
@st.cache_resource
def createWrapper():
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    wrapper = DuckDuckGoSearchAPIWrapper(region='us-en', time="y", max_results=10) #time parameter Options: d, w, m, y
    return wrapper

loadDictionary()
tokencounter = tokencounter()

if "wrapper" not in st.session_state:
    st.session_state.wrapper = createWrapper()
if "searchquery" not in st.session_state:
    st.session_state.searchquery = ''
if "limiter" not in st.session_state:
    st.session_state.limiter = 0    

def getWebDocuments(q,n):
    #wrapper = createWrapper()
    console.print(f'[bold red1]Searching for {q}')
    console.print(90*'=')
    rawdb = st.session_state.wrapper.results(q,max_results=n)
    docdocs = []
    mt = 1
    numofdocs = len(rawdb)
    for items in rawdb:
        url = items["link"]
        try:  #useful if the url is no reachable
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            kw = []
            #we merge nltk keywords and meta webpage keywords
            for i in article.keywords+article.meta_keywords:
                if i == '': #no blck yeywords for us
                    pass
                else:
                    kw.append(i)
            if article.text == '': #sometimes there is no text to parse. so we use the snippet
                docdocs.append(Document(page_content = items["snippet"], metadata = {
                    'source': items["link"],
                    'title': items["title"],
                    'snippet': items["snippet"],
                    'author':article.authors,
                    'keywords':kw,
                    'meta_description':article.meta_description,
                    'meta_img':article.meta_img,
                    'top_image':article.top_image,
                    'publish_date':article.publish_date,
                    'summary':article.summary}))
            else:
                docdocs.append(Document(page_content = article.text.replace('\n\n',''), metadata = {
                    'source': items["link"],
                    'title': items["title"],
                    'snippet': items["snippet"],
                    'author':article.authors,
                    'keywords':kw,
                    'meta_description':article.meta_description,
                    'meta_img':article.meta_img,
                    'top_image':article.top_image,
                    'publish_date':article.publish_date,
                    'summary':article.summary}))
            console.print(f'Prepared Document n.{mt} out of {numofdocs}')
            mt +=1      
        except:
            pass    
    st.session_state.searches.append({'role': 'results','results': docdocs})



st.title("AI powered Web Document Search")
st.write('Using Newspaper3k and DuckDuckGo LangChain wrapper')


def savelog(db):
        log = []
        finaltext = ''
        for section in db:
            singlesearch = ''
            #console.print(section)
            if section['role'] == 'query':
                header1 = f"##### 🪄 Query: *{section['query']}*"
                singlesearch = singlesearch + header1 + '\n'
            else:
                header2 = '##### 🗃️ Web Documents'
                singlesearch = singlesearch + header2 + '\n'
                for items in section['results']:
                    #console.print(section['results'])
                    singlesearch = singlesearch + '📝 ' + f"{items.metadata['title']}\n"
                    singlesearch = singlesearch + items.page_content + '\n--\n'
                    singlesearch = singlesearch + f"""'source': {items.metadata['source']}
'title': {items.metadata['title']}
'snippet': {items.metadata['snippet']}
'author':{items.metadata['author']}
'keywords':{items.metadata['keywords']}
'meta_description':{items.metadata['meta_description']}
'meta_img':{items.metadata['meta_img']}
'top_image':{items.metadata['top_image']}
'publish_date':{items.metadata['publish_date']}
'summary':{items.metadata['summary']}
------------------------//---------------------------\n"""
            log.append(singlesearch)
        for i in log:
            finaltext = finaltext + i
        writehistory(st.session_state.sessionlogfilename,finaltext)

def savedb(db):
    output = open(st.session_state.sessionDBfilename, 'wb')
    pickle.dump(db, output)
    output.close()


# CREATE THE SIDEBAR
with st.sidebar:
    st.image('logoDDG.png', use_column_width=True)
    st.session_state.limiter = st.slider('N.of Docs:', min_value=1, max_value=5, value=3, step=1)
    st.markdown(f"**_DB name**: {st.session_state.sessionDBfilename}")
    st.markdown(f"**___Logfile**: {st.session_state.sessionlogfilename}")
    logevents = st.empty()
    dbevents = st.empty()
    btnSAVE = st.button("Save DB",type="primary", use_container_width=True)

# CREATE SEARCH FORM
with st.form("my_form"):
    st.image('searchddglogo.png',)
    query = st.text_input(
        "What do you want to search?",
        placeholder="your query here",
        label_visibility='collapsed')
    # Every form must have a submit button.
    submitted = st.form_submit_button("Search")
    if submitted:
        with st.spinner(text="Search in progress..."):
            st.session_state.searchquery = query
            st.session_state.searches.append({'role' : 'query','query': query})
            getWebDocuments(query,st.session_state.limiter)
            savelog(st.session_state.searches)
        #logevents.success('Search saved')
        st.toast('Search saved!', icon='🎉')

st.markdown('### AI serach results')
with st.container(height=500, border=True):
    if len(st.session_state.searches) >0:
        for section in st.session_state.searches:
            #console.print(section)
            if section['role'] == 'query':
                st.write(f"##### 🪄 Query: *{section['query']}*")
            else:
                with st.container():
                    st.markdown('##### 🗃️ Web Documents')
                    for items in section['results']:
                        #console.print(section['results'])
                        with st.expander(label=items.metadata['title'], expanded=False, icon='📝'):
                            st.image(items.metadata['top_image'], width=350)
                            st.write(f"source url: {items.metadata['source']}\n\nSummary: {items.metadata['summary']}")
                            st.write(items.page_content)
                            st.divider()
                            st.write(items.metadata)
                st.divider()                
if btnSAVE:
    savedb(st.session_state.searches)
    #dbevents.success('dB saved')
    st.toast('dB saved!', icon='🎉')

console.print(Markdown("> SEARCH COMPLETED..."))
console.print(" - ")


