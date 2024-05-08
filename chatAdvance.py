# %%
import os
import re
import json

# To help construct our Chat Messages
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# We will be using ChatGPT model (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

os.environ["OPENAI_API_KEY"] = "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"

# %%
# from langchain_community.document_loaders import UnstructuredFileLoader
# loader = UnstructuredFileLoader("/Users/alphatech/Downloads/cnv_2024_04_08_3bd653f30b35923f0168g/cnv_2024_04_08_3bd653f30b35923f0168g.tex")
# docs = loader.load()

# %%
# import os
# import PyPDF2
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.vectorstores import FAISS

# def extract_text_from_pdf(pdf_file_path):
    
#     try:
#         loader = PyPDFLoader(pdf_file_path)
#         data = loader.load()
    
#     except Exception as e:
#         print(f"Error reading PDF file '{pdf_file_path}': {e}")
#     return data



# def fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name):
#     exam_path = os.path.join(root_folder, exam_name)
#     topic_path = os.path.join(exam_path, topic_name)
#     chapter_path = os.path.join(topic_path, chapter_name)
#     subchapter_path = os.path.join(chapter_path, subchapter_name)

#     pdf_contents = []

#     for subchapter, _, pdf_files in os.walk(subchapter_path):
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(subchapter, pdf_file)
#             print(f"Processing: {pdf_path}")
#             if pdf_file.endswith('.pdf'):  # Check if the file is a PDF
#                 pdf_content = extract_text_from_pdf(pdf_path)
#                 if pdf_content:
#                     pdf_contents.extend(pdf_content)
#                     print(f"Successfully extracted content from: {pdf_path}")
#                 else:
#                     print(f"Failed to extract content from: {pdf_path}")
#             else:
#                 print(f"Skipping non-PDF file: {pdf_path}")
            
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         pdf_texts = text_splitter.split_documents(pdf_contents)
#     return pdf_texts

# # Example usage:
# root_folder = '/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1'
# exam_name = "CAIA Level 1"
# topic_name = "Hedge Funds"
# chapter_name = "5.2 Macro and Managed Futures Funds"
# subchapter_name = "Systematic Trading"

# pdf_contents = fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name)


# %%
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# loader = PyPDFLoader('/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1')
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(data)

# %%
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
#from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader

app = Flask(__name__)
#run_with_ngrok(app)  # Initialize ngrok when the app is run

CORS(app, supports_credentials=True, allow_headers=["Content-Type"])

def extract_text_from_pdf(file_path):
            
            try:
                loader = Docx2txtLoader(file_path)
                docx_file = loader.load()
                print("Docx file is: ", docx_file)
                return docx_file
            except Exception as e:
                print(f"Error reading docx file '{file_path}': {e}")

def fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name):
    exam_path = os.path.join(root_folder, exam_name)
    topic_path = os.path.join(exam_path, topic_name)
    chapter_path = os.path.join(topic_path, chapter_name)
    subchapter_path = os.path.join(chapter_path, subchapter_name)

    pdf_texts = []
    pdf_contents = []

    for subchapter, _, pdf_files in os.walk(subchapter_path):
        for pdf_file in pdf_files:
            pdf_path = os.path.join(subchapter, pdf_file)
            print(f"Processing: {pdf_path}")
            if pdf_file.endswith('.pdf'):  # Check if the file is a PDF
                pdf_content = extract_text_from_pdf(pdf_path)
                if pdf_content:
                    pdf_contents.extend(pdf_content)
                    print(f"Successfully extracted content from: {pdf_path}")
                else:
                    print(f"Failed to extract content from: {pdf_path}")
            else:
                print(f"Skipping non-Docx file: {pdf_path}")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_texts = text_splitter.split_documents(pdf_contents)
    return pdf_texts
        

@app.route('/', methods=['GET', 'POST'])
def ask_question():
    try:
        data = request.get_json()
        
        question = data['question']
        
        chapter = data['subChapter']
        print("chapter is: ", chapter)
        
        topic = data['chapter']
        print("topic is: ", topic)
        
        subChapter = data['topic']
        print("subchapter is: ", subChapter)
        
        if question is None or question.strip() == '':
                raise ValueError("Invalid or empty question")
        
        root_folder = '/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1'
        exam_name = "CAIA Level 1"
        
        pdf_contents = fetch_pdfs(root_folder, exam_name, topic, chapter, subChapter)

        
        chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Ensure the answer is correct and its explaination to be conforming to the context.

        {context}

        Question: {question}

        Helpful Answer:"""
        prompt = PromptTemplate.from_template(template)

        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        vectorstore = FAISS.from_documents(pdf_contents, embedding=embeddings)
        
        retriever = vectorstore.as_retriever()
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | chat_model
            | StrOutputParser()
        )

        ans = chain.invoke(question)
        print(ans)
    
        if ans is not None:
            return jsonify({"result": ans})
        else:
            return jsonify({"result": "No answer found"})
    except Exception as e:
        print("Exception:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)



