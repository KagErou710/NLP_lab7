import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import torch
from langchain import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory

from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain_community.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
)

from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
prompt_template = """
    I'm your friendly chatbot named AIT-GPT, here to assist you with any questions you have about AIT. 
    If you are interesting about Background of AIT, Programs, Degrees, Admissions, Tuitions fee, Ongoing Research, anything about AIT, please feel free to ask.
    I am covering everything in official web page. So if you felt lazy to search in web page by yourself, no hesitate to asking me.
    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)
PROMPT.format(
    context = "I heard you want to go to AIT",
    question = "What is AIT"
)

nlp_docs = r'C:\Users\Tairo Kageyama\Documents\GitHub\Python-fo-Natural-Language-Processing-main\lab7\Data\AIT_web_reduced.pdf'

loader = PyMuPDFLoader(nlp_docs)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100
)

doc = text_splitter.split_documents(documents)

model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : device}
)

vector_path = '../vector-store'
if not os.path.exists(vector_path):
    os.makedirs(vector_path)
    print('create path done')

vectordb = FAISS.from_documents(
    documents = doc,
    embedding = embedding_model
)

db_file_name = 'nlp_stanford'

vectordb.save_local(
    folder_path = os.path.join(vector_path, db_file_name),
    index_name = 'nlp' #default index
)

vector_path = '../vector-store'
db_file_name = 'nlp_stanford'

from langchain.vectorstores import FAISS

vectordb = FAISS.load_local(
    folder_path = os.path.join(vector_path, db_file_name),
    embeddings = embedding_model,
    index_name = 'nlp', #default index
    # allow_dangerous_deserialization=True
)   

retriever = vectordb.as_retriever()

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({'input':'hi'}, {'output':'What\'s up?'})
memory.save_context({"input":'How are you?'},{'output': 'I\'m quite good. How about you?'})
memory.load_memory_variables({})

#chain

model_id = r'C:\Users\Tairo Kageyama\Documents\GitHub\Python-fo-Natural-Language-Processing-main\lab7\models\fastchat-t5-3b-v1.0'
tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id
bitsandbyte_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    # quantization_config = bitsandbyte_config, #caution Nvidia
    # device_map = 'auto',
    load_in_8bit = False
)
pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 1024,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)

def answerQn(prompt_question):
    answer = chain({"question" : prompt_question})
    ans = answer['answer']
    # ans = 'Hello'
    return ans

# ----------------------------------------------------------------------------------------


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AIT-GPT"),
    html.H3("""I'm your friendly chatbot named AIT-GPT, here to assist you with any questions you have about AIT. \n
    If you are interesting about Background of AIT, Programs, Degrees, Admissions, Tuitions fee, Ongoing Research, anything about AIT, please feel free to ask.\n
    I am covering everything in official web page. So if you felt lazy to search in web page by yourself, no hesitate to asking me."""),
    dcc.Input(id='user_input', type='text', placeholder='Enter a question'),
    # dcc.Input(id='query', type='text', placeholder='Enter a word'),
    html.Button('Ask!!', id='process_button'),
    html.Div(id='output_div')
])

@app.callback(
    Output('output_div', 'children'),
    [Input('process_button', 'n_clicks')],
    [dash.dependencies.State('user_input', 'value')]
)
def process_word(n_clicks, user_input):
    if n_clicks is not None and n_clicks > 0:
        output = answerQn(user_input) #here
        return [
            html.P(f'You entered: {user_input}'),
            html.P(f'Answer : {output}'),
        ]
    

if __name__ == '__main__':
    app.run_server(debug=True)