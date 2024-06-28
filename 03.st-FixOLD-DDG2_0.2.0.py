
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
import random
import string
import datetime

st.set_page_config(page_title="AI powered web serach", layout="wide",page_icon='📱')

# to write out a log file PLAIN TXT - this will rewrite every time the file
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

# Initialize search history and dataDB / the one to be saved
if "searches" not in st.session_state:
    st.session_state.searches = []
if "dataDB" not in st.session_state:
    st.session_state.dataDB = []

# Newsletter3k utility disctionary from NLTK
@st.cache_resource
def loadDictionary():
    import nltk
    nltk.download('punkt')

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
if "wrapper" not in st.session_state:
    st.session_state.wrapper = createWrapper()
if "searchquery" not in st.session_state:
    st.session_state.searchquery = ''
if "limiter" not in st.session_state:
    st.session_state.limiter = 0    
if "chatdocs" not in st.session_state:
    st.session_state.chatdocs = 0   
if "uploadedDoc" not in st.session_state:
    st.session_state.uploadedDoc = [] 
if "uploadedText" not in st.session_state:
    st.session_state.uploadedText = ''     
if "fileloaded" not in st.session_state:
    st.session_state.fileloaded = 1  

def LoadDocuments(pickleDB):
    mt = 1
    numofdocs = len(pickleDB)
    for items in pickleDB:
        st.session_state.dataDB.append(Document(page_content = items.page_content, metadata = {
            'search_query' : items.metadata['search_query'],
            'source': items.metadata['source'],
            'title': items.metadata['title'],
            'snippet': items.metadata['snippet'],
            'author':items.metadata['author'],
            'keywords':items.metadata['keywords'],
            'meta_description':items.metadata['meta_description'],
            'meta_img':items.metadata['meta_img'],
            'top_image':items.metadata['top_image'],
            'publish_date':items.metadata['publish_date'],
            'summary':items.metadata['summary']}))           
        console.print(f'Prepared Document n.{mt} out of {numofdocs}')
        mt +=1      


if "keyDDGfile" not in st.session_state:
    st.session_state.keyDDGfile = 1  

def resetall():
    st.session_state.keyimagefile += 1
    st.rerun()  

#################### MAIN STREaMLIT APP ##################################
st.title("AI powered Web Document Search")
st.write('Using Newspaper3k and DuckDuckGo LangChain wrapper')
st.divider()
# Upload the audio file
file1 = st.sidebar.file_uploader("Upload a text document", 
                                    type=["ddg"],accept_multiple_files=False, 
                                    key=st.session_state.keyDDGfile)
st.divider()
# Save the searches into plain TXT file
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

def load(fname):
    console.print(f'[blink2 orange1]Loading...')
    pkl_file = open(fname.name, 'rb')
    data_docs = pickle.load(pkl_file)
    pkl_file.close()
    st.session_state.dataDB = data_docs

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
    btnLOAD = st.button("Load OLD DB",type="primary", use_container_width=True,disabled=st.session_state.keyDDGfile)
    st.divider()
    reset_btn = st.button('🧻✨ **Reset Document** ', type='primary')
    btnSAVE = st.button("Save DB",type="secondary", use_container_width=True, disabled=st.session_state.fileloaded)

if reset_btn:
    resetall()
    try:
        st.session_state.uploadedDoc = [] 
        st.session_state.uploadedText = '' 
        st.session_state.fileloaded = 1
        st.session_state.keyDDGfile = 1
    except:
        pass

if file1:
    st.session_state.keyDDGfile = 0

if btnLOAD:
    load(file1)
    st.session_state.fileloaded = 0

if btnSAVE:
    savelog()
    savedb(st.session_state.searches)
    #dbevents.success('dB saved')
    st.toast('dB saved!', icon='🎉')    


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


console.print(Markdown("> SEARCH COMPLETED..."))
console.print(" - ")


