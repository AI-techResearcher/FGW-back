
# %%
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# We will be using ChatGPT model (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI

# To parse outputs and get structured data back
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from flask_cors import CORS
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

def extract_text_from_pdf(file_path):
    
    try:
        loader = TextLoader(file_path)
        docx_file = loader.load()
        print("tex file is: ", file_path)
        return docx_file
    
    except Exception as e:
        print(f"Error reading tex file '{file_path}': {e}")
        return " "



def extract_text_from_all_pdfs(root_folder):
    docs = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                data = extract_text_from_pdf(file_path)
                docs.extend(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_texts = text_splitter.split_documents(docs)
    return pdf_texts

# %%
os.environ["OPENAI_API_KEY"] = "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"

chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')

from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# %%
app = Flask(__name__)
CORS(app, supports_credentials=True, allow_headers=["Content-Type"])  # Enable CORS for all routes

@app.route('/', methods=['GET','POST'])
def ask_question():
    try:
        data = request.get_json()
        
        root_folder = '/Users/alphatech/Desktop/FGW_latexData'
        exam_name = "CAIA Level 1"
        root_folder = os.path.join(root_folder, exam_name)
        
        pdf_texts = extract_text_from_all_pdfs(root_folder)
        
        vectorstore = FAISS.from_documents(pdf_texts, embedding=embeddings)
        
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=chat_model, chain_type='stuff',
            retriever=vectorstore.as_retriever()
        )
        
        question = data.get('question')
        print(question)
        if question is None or question.strip() == '':
                raise ValueError("Invalid or empty question")
            
        result = rag_pipeline(question)
        print(result)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if os.environ["ENVIRONMENT"] == "production":
        serve(app, listen='*:5000')
    else:
        app.run(port = 5000)


