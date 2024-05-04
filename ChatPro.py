# %%
# !pip install openai
# !pip install tiktoken
# !pip install faiss-cpu

# !pip install langchain_experimental
# !pip install "langchain[docarray]"

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import langchain

langchain.debug = True

# %%
# import os
# import PyPDF2

# def extract_pdf_content(pdf_path):
#     try:
#         with open(pdf_path, 'rb') as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page_num in range(len(pdf_reader.pages)):
#                 text += pdf_reader.pages[page_num].extract_text()
#             return text
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return ""

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
#             pdf_content = extract_pdf_content(pdf_path)
#             if pdf_content:
#                 pdf_contents.append(pdf_content)
#                 print(f"Successfully extracted content from: {pdf_path}")
#             else:
#                 print(f"Failed to extract content from: {pdf_path}")

#     return pdf_contents

# # Example usage:
# root_folder = "/Users/alphatech/Downloads/FGW_Data_sampleTheory/CAIA Level 1"
# exam_name = "CAIA Level 1"
# topic_name = "Hedge Funds"
# chapter_name = "5.5 Funds of Hedge Funds"
# subchapter_name = "Investing in Funds of Hedge Funds"

# pdf_contents = fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name)

# if pdf_contents:
#     for i, content in enumerate(pdf_contents, start=1):
#         print(f"PDF Document {i} Content:")
#         print(content)
#         print("=" * 50)
# else:
#     print("No PDF content found.")


# %%
# from langchain.chains import (
#     StuffDocumentsChain,
#     LLMChain,
#     ReduceDocumentsChain,
#     MapReduceDocumentsChain,
# )
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import OpenAI

# # This controls how each document will be formatted. Specifically,
# # it will be passed to `format_document` - see that function for more
# # details.
# document_prompt = PromptTemplate(
#     input_variables=["page_content"],
#      template="{page_content}"
# )

# document_variable_name = "text"

# # The prompt here should take as an input variable the
# # `document_variable_name`
# prompt = PromptTemplate.from_template(
#     "Summarize this content: {text}"
# )
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # We now define how to combine these summaries
# reduce_prompt = PromptTemplate.from_template(
#     "Combine these summaries: {text}"
# )
# reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
# combine_documents_chain = StuffDocumentsChain(
#     llm_chain=reduce_llm_chain,
#     document_prompt=document_prompt,
#     document_variable_name=document_variable_name
# )
# reduce_documents_chain = ReduceDocumentsChain(
#     combine_documents_chain=combine_documents_chain,
# )
# chain = MapReduceDocumentsChain(
#     llm_chain=llm_chain,
#     reduce_documents_chain=reduce_documents_chain,
# )

# %%
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# loader = PyPDFLoader("/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1/CAIA Level 1/Hedge Funds/5.2 Macro and Managed Futures Funds/Systematic Trading/Learning Objective.docx")
# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# text = text_splitter.split_documents(data)

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
chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')

# %%
# import os
# import PyPDF2
# from docx import Document

# def extract_text_from_docx(docx_file):
#     document = Document(docx_file)
#     text = ""
#     for paragraph in document.paragraphs:
#         text += paragraph.text + "\n"
#     return text

# # Replace 'your_file.docx' with the path to your Word document
# docx_file = "/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1/CAIA Level 1/Hedge Funds/5.2 Macro and Managed Futures Funds/Systematic Trading/Learning Objective.docx"
# lo = extract_text_from_docx(docx_file)
# print(lo)       

# %%
# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def extract_pdf_content(pdf_path):
#     try:      
#         loader = PyPDFLoader(pdf_path)
#         pdf_doc = loader.load()
#         print("pdf doc is: ", pdf_doc)
#         return pdf_doc
    
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return ""

    
# def fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name):
    
#     exam_path = os.path.join(root_folder, exam_name)
#     topic_path = os.path.join(exam_path, topic_name)
#     chapter_path = os.path.join(topic_path, chapter_name)
#     subchapter_path = os.path.join(chapter_path, subchapter_name)

#     pdf_texts = []
#     pdf_contents = []

#     for subchapter, _, pdf_files in os.walk(subchapter_path):
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(subchapter, pdf_file)
#             print(f"Processing: {pdf_path}")
            
#             if pdf_file.endswith('.pdf'):  # Check if the file is a PDF
#                 pdf_content = extract_pdf_content(pdf_path)
                
#                 if pdf_content:
#                     pdf_contents.extend(pdf_content)
#                     print(f"Successfully extracted content from: {pdf_path}")
#                 else:
#                     print(f"Failed to extract content from: {pdf_path}")
                    
#             else:
#                 print(f"Skipping non-PDF file: {pdf_path}")
            
#         # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         # pdf_texts = text_splitter.split_documents(pdf_contents)
#         #print("pdf texts in fetch_pdfs: ", pdf_texts)
#     return pdf_contents    

# root_folder = "/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1"
# exam_name = "CAIA Level 1"
# topic_name = "Hedge Funds"
# chapter_name = "5.2 Macro and Managed Futures Funds"
# subchapter_name = "Systematic Trading"

# docs = fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name)

# %%
# text = ""
# for paragraph in docs:
#     text += paragraph.page_content + "\n"
# print(text)

# %%
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Rpdf_texts = text_splitter.split_documents(docs)

# %%
#Rpdf_texts

# %%
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# model_name = "BAAI/bge-small-en"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# hf = HuggingFaceBgeEmbeddings(
#         model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )
# vectorstore = FAISS.from_documents(Rpdf_texts, embedding=hf)

# retriever = vectorstore.as_retriever()

# from langchain.chat_models import ChatOpenAI
# from langchain.retrievers.multi_query import MultiQueryRetriever

# # we instantiated the retreiever above
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=retriever, llm=chat_model
# )

# # Set logging for the queries
# import logging

# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# unique_docs = retriever_from_llm.get_relevant_documents(query=lo)

# %%
# from langchain_text_splitters import CharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# pdf_texts = text_splitter.split_documents(docs)

# %%


# %%
# from langchain import PromptTemplate
# from langchain.chains.summarize import load_summarize_chain

# map_prompt_template = """
#                         Write a summary of this chunk of text that includes the main points and any important details.
#                         {text}
#                         """

# map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

# combine_prompt_template = """
#                     Write a concise summary of the following text delimited by triple backquotes.
#                     Return your response in bullet points which covers the key points of the text.
#                     ```{text}```
#                     BULLET POINT SUMMARY:
#                     """

# combine_prompt = PromptTemplate(
#     template=combine_prompt_template, input_variables=["text"]
# )

# map_reduce_chain = load_summarize_chain(
#     chat_model,
#     chain_type="map_reduce",
#     map_prompt=map_prompt,
#     combine_prompt=combine_prompt,
#     return_intermediate_steps=True,
# )

# summarizedtext = map_reduce_chain({"input_documents": Rpdf_texts})



# %%
# Rtext = Rsummarizedtext["intermediate_steps"]
# Rtext

# %%
# Rtext = summarizedtext["output_text"]
#Rtext

# %%
# from langchain import PromptTemplate
# from langchain.chains.summarize import load_summarize_chain

# map_prompt_template = """
#                         Write a summary of this chunk of text that includes the main points and any important details.
#                         {text}
#                         """

# map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

# combine_prompt_template = """
#                     Write a concise summary of the following text delimited by triple backquotes.
#                     Return your response in bullet points which covers the key points of the text.
#                     ```{text}```
#                     BULLET POINT SUMMARY:
#                     """

# combine_prompt = PromptTemplate(
#     template=combine_prompt_template, input_variables=["text"]
# )

# map_reduce_chain = load_summarize_chain(
#     chat_model,
#     chain_type="map_reduce",
#     map_prompt=map_prompt,
#     combine_prompt=combine_prompt,
#     return_intermediate_steps=True,
# )

# summarizedtext = map_reduce_chain({"input_documents": pdf_texts})


# %%
# out_text = summarizedtext["output_text"]
# text = summarizedtext["intermediate_steps"]
# print("intermediate: ",text)
# #print("out: ",out_text)
# text

# %%
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

def generate(question_type, docs, lo, number, difficulty):

    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(docs, embedding=hf)

    retriever = vectorstore.as_retriever()

    # we instantiated the retreiever above
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=chat_model
    )

    # Set logging for the queries
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    text = retriever_from_llm.get_relevant_documents(query=lo)
    
    is_case_study = "Case study" in question_type
    print(is_case_study)
    
    if is_case_study:

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("""You are a helpfull writing assistant.
Given the text delimited by triple backticks, it is your job to write a {question_type} (Case study is an in-depth and detailed examination of the given text within a real-world context) with difficulty level {difficulty}.

Case study: [write a case study here]
Question number:

Ensure to make {number} questions from the case study.
Do not include answers of the questions in the quiz.
Ensure the case study to be conforming to the text.
Make sure the questions are not repeated.
          \n```{text}``` """)
            ],
            input_variables=["text", "question_type", "difficulty","number"],  # Fix the order of variables
            #partial_variables={"format_instructions": format_instructions}
        )

        user_query = prompt.format_prompt(text=text, question_type=question_type, difficulty=difficulty, number=number)  # Adjust values accordingly
        response = chat_model(user_query.to_messages())

    else:

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("""you are an expert financial analyst:
Given the text delimited by triple backticks, it is your job to create a quiz of {number} {question_type} with difficulty level {difficulty}.

Question number:

Do not include answers of the questions in the quiz.
Make sure that questions are not repeated and check all the questions to be conforming to the text as well.
Ensure to make the {number} questions.
          \n```{text}``` """)
            ],
            input_variables=["text", "number", "question_type", 'difficulty'],  # Fix the order of variables
            #partial_variables={"format_instructions": format_instructions}
        )

        user_query = prompt.format_prompt(text = text, number=number, question_type=question_type, difficulty=difficulty)  # Adjust values accordingly
        response = chat_model(user_query.to_messages())
        
    return response

#Example usage:
# question_type = "open-ended questions"

# response = generate(question_type, Rpdf_texts, lo, number=2, difficulty = "medium")


# %%
#print(response.content)

# %%
# response = response.content
# # Split the response into individual questions
# questions = response.split('\n\n')

# # Remove empty strings from the list
# questions = [question.strip() for question in questions if question.strip()]

# print(questions)

# %%
# response = response
# print(response)

# %%
#questions

# %%
# for question in questions:
#     print(question)
# question

# %% [markdown]
# Answering module

# %%
# prompt = ChatPromptTemplate(
#             messages=[
#                 HumanMessagePromptTemplate.from_template(""" You are a knowledgeable assistant capable of answering multiple-choice questions {questions} using the provided text document for reference.
# Given the text delimited by triple backticks, it is your job to answer the following multiple choice questions with explanation of the correct answer from the text.

# Question number:
# "[Insert your multiple-choice question here]"
# A) [Option A]
# B) [Option B]
# C) [Option C]
# D) [Option D]
# Answer: [correct option here]
# Explaination: [explain and solidify your answer here]
# Ensure the answer and its explaination to be conforming to the text.

#           \n```{text}``` """)
#             ],
#             input_variables=["questions", "text"],  # Fix the order of variables
#             #partial_variables={"format_instructions": format_instructions}
#         )



# %% [markdown]
# 

# %% [markdown]
# Answers to the questions

# %%
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

def answers_response(question, question_type, data):
        
        is_case_study = "case study" in question_type
        print(is_case_study)
        
        if is_case_study:
                template = """ You are a knowledgeable assistant capable of answering a case study based multiple choice question {question} from the case study in the question for reference.
                
                Question number: [only the question number]
                Answer: [only correct answer here]
                Explaination: [explain and solidify your answer here]
                
                You can also take help from the given the context delimited by triple backticks, to ensure the correct answer and its explanation is conforming to the case study.
                Ensure the correct answer and its explaination to be conforming to the case study in the {question}.
                Do not repeat the question and each question should be answered only once.
                

                        \n```{context}``` """
                        
        else:        
                template = """ You are a knowledgeable assistant capable of answering a multiple-choice question {question} using the provided context for reference.
                Given the context delimited by triple backticks, it is your job to answer the following multiple choice questions with explanation of the correct answer from the context.
                Question number: [only the question number]
                Answer: [only correct option here]
                Explaination: [explain and solidify your answer here]
                If no multiple choices are given in the question, answer only the the open-ended questions as given in the question.
                Ensure the correct answer and its explaination to be conforming to the context.
                

                        \n```{context}``` """
        
        prompt = ChatPromptTemplate.from_template(template)
        

        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        
        vectorstore = FAISS.from_documents(data, embedding=hf)
        retriever = vectorstore.as_retriever()
        
        chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
        )
        
        answer = chain.invoke(question)
               
        return answer

#ans_result = answers_response(question_type)
#prompt = ChatPromptTemplate.from_template(template)

# %%
# template = """ You are a knowledgeable assistant capable of answering multiple-choice question {question} from the context provided for reference.
                
#                 Question number: [only the question number]
#                 Answer: [only correct answer here]
#                 Explaination: [explain and solidify your answer here]
                
#                 Ensure the correct answer and its explaination to be conforming to the case study in the {question}.
#                 Do not repeat the question and each question should be answered only once.
                

#                         \n```{context}``` """
# prompt = ChatPromptTemplate.from_template(template)

# %%
# answers = []
# for question in questions:
#     answer = chain.invoke(question)
#     answers.append(answer)

# %%
# for ans in answers:
#   print(ans)

# %%
#answers

# %%
# template = """ You are a knowledgeable assistant capable of answering multiple-choice questions {question} using the provided text document for reference.
# Given the context delimited by triple backticks, it is your job to answer the following multiple choice questions with explanation of the correct answer from the text.
# Question number:
# Answer: [correct option here]
# Explaination: [explain and solidify your answer here]
# Ensure the correct answer and its explaination to be conforming to the context.

#           \n```{summaries}``` """
          
# prompt = ChatPromptTemplate.from_template(template)

# %%
# answer_generation_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=chat_model,
#                                                 chain_type="stuff",
#                                                 retriever=vectorstore.as_retriever(),
#                                                 chain_type_kwargs={
#                                                 "prompt": prompt,
                                                
#                                                 },)

# %%

# for question in questions_list:
#     answer = answer_generation_chain.invoke(question)

# %% [markdown]
# Math Q&A

# %%
# import aiohttp
# async with aiohttp.ClientSession() as session:
#         async with session.get("http://python.org",
#                            proxy="http://proxy.com") as resp:
#                 print(resp.status)

# %%
# root_folder = '/Users/alphatech/Desktop/Educational web app/fgwpro-main2'
# exam_name = "CAIA Level 1"
# topic_name = "Hedge Funds"
# chapter_name = "5.1 Structure of the Hedge Fund Industry"
# subchapter_name = "Hedge Fund"
# doc = "Math.docx"
# file_path = os.path.join(root_folder, exam_name, topic_name, chapter_name, subchapter_name, doc)
# if not os.path.exists(file_path):
#         print("No Math!")
        


# %% [markdown]
# Math questions

# %%
from langchain_community.document_loaders import Docx2txtLoader

def math_generate(question_type, file_path, difficulty, math_number):
    if not os.path.exists(file_path):
        print("No Math!")
        return None

    loader = Docx2txtLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text = text_splitter.split_documents(data)
    
    is_case_study = "case study" in question_type
    print(is_case_study)
    if is_case_study:

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(""" You are a helpfull assistant.
Given the questions and answers with explaination in the given text delimited by triple backticks, it is your job to write a {question_type} with difficulty level {difficulty}.

Case study:
Question number:

Ensure to write a case study based on the given text.
Ensure the case study to be conforming to the text.
Ensure to make {math_number} questions from the case study.
Do not include answers of the questions in the quiz.
Make sure the questions are not repeated.
          \n```{text}``` """)
            ],
            input_variables=["text", "question_type", "difficulty", "math_number"],  # Fix the order of variables
            #partial_variables={"format_instructions": format_instructions}
        )

        user_query = prompt.format_prompt(text=text, question_type=question_type, difficulty=difficulty, math_number=math_number)  # Adjust values accordingly
        math_questions = chat_model(user_query.to_messages())

    else:

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("""you are an expert financial analyst:
Given the Questions, answers with explaination in the given text delimited by triple backticks, it is your job to create a mathematical quiz of {math_number} {question_type} with difficulty level {difficulty}.

Math Quiz:
Question number:

Do not include answers of the questions in the quiz.
Make sure that questions are not repeated and check all the questions to be conforming to the text as well.
Ensure to make the {math_number} questions.
          \n```{text}``` """)
            ],
            input_variables=["text", "math_number", "question_type", "difficulty"],  # Fix the order of variables
            #partial_variables={"format_instructions": format_instructions}
        )

        user_query = prompt.format_prompt(text=text, math_number=math_number, question_type=question_type, difficulty=difficulty)  # Adjust values accordingly
        math_questions = chat_model(user_query.to_messages())


    return math_questions

# #Example usage:
# question_type = "open-ended questions"

# math_response = math_generate(question_type, file_path, difficulty="medium", math_number=2)


# %%
# math_response = math_response.content
# # Split the response into individual questions
# questions = math_response.split('\n\n')

# # Remove empty strings from the list
# math_questions = [question.strip() for question in questions if question.strip()]

# print(math_questions)

# %%
# math_caseStudy = math_questions[0:2]

# %%
# for question in math_questions[2:]:
#     print(question)

# %%
# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("/Users/alphatech/Downloads/FGW Data - sample/CAIA Level 1/CAIA Level 1/Hedge Funds/5.2 Macro and Managed Futures Funds/Systematic Trading/math.pdf")
# math_data = loader.load()

# %%
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# math_data = text_splitter.split_documents(math_data)

# %% [markdown]
# Math answers

# %%
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS

def math_answers(file_path, math_question):
    if math_question:
        loader = Docx2txtLoader(file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        math_data = text_splitter.split_documents(data)
        
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        math_hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        math_vectorstore = FAISS.from_documents(math_data, embedding=math_hf)
        
        template = """ You are a knowledgeable assistant capable of answering multiple-choice question {math_question} using the provided context for reference.
        Given the context delimited by triple backticks, it is your job to answer the given multiple choice question with explaination of the correct answer using the context.

        Math Question number: [only question number here]
        Answer: [correct option here]
        Explaination: [explain and solidify your answer here]

        Make sure to answer only the given question {math_question}.
        Ensure the correct answer and its explaination to be conforming to the context.

                \n```{context}``` """
                
        math_prompt = ChatPromptTemplate.from_template(template)
        

        math_retriever = math_vectorstore.as_retriever()

        math_chain = (
            {"context": math_retriever, "math_question": RunnablePassthrough()}
            | math_prompt
            | chat_model
            | StrOutputParser()
        )
        
        answer = math_chain.invoke(math_question)
        
        return answer
    
    else:
        return print("No Math!")
    
    

# %%
# math_answers = []
# for question in math_questions[2:]:
#     answer = math_chain.invoke(question)
#     math_answers.append(answer)
 

# %%
# for ans in math_answers:
#   print(ans)

# %%
# root_folder = '/Users/alphatech/Desktop/Educational web app/fgwpro-main2/DataTheory/CAIA Level 1'
# exam_name = "CAIA Level 1"
# topic_name = "Hedge Funds"
# chapter_name = "5.2 Macro and Managed Futures Funds"
# subchapter_name = "Systematic Trading"

# file_path = os.path.join(root_folder, exam_name, topic_name, chapter_name, subchapter_name)
# file_path

# %%
# def extract_pdf_content(pdf_path):
#     try:      
#         loader = Docx2txtLoader(pdf_path)
#         docx_file = loader.load()
#         print("Docx file is: ", docx_file)
#         return docx_file
    
#     except Exception as e:
#         print(f"Error extracting text from {pdf_path}: {e}")
#         return ""

    
# def fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name):
    
#     exam_path = os.path.join(root_folder, exam_name)
#     topic_path = os.path.join(exam_path, topic_name)
#     chapter_path = os.path.join(topic_path, chapter_name)
#     subchapter_path = os.path.join(chapter_path, subchapter_name)

#     #pdf_texts = []
#     pdf_contents = []

#     for subchapter, _, pdf_files in os.walk(subchapter_path):
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(subchapter, pdf_file)
#             print(f"Processing: {pdf_path}")
            
#             if pdf_file.endswith('.docx'):  # Check if the file is a PDF
#                 pdf_content = extract_pdf_content(pdf_path)
                
#                 if pdf_content:
#                     pdf_contents.extend(pdf_content)
#                     print(f"Successfully extracted content from: {pdf_path}")
#                 else:
#                     print(f"Failed to extract content from: {pdf_path}")
                    
#             else:
#                 print(f"Skipping non-PDF file: {pdf_path}")
#     from langchain_text_splitters import CharacterTextSplitter
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     pdf_texts = text_splitter.split_documents(pdf_contents)
        
#     return pdf_texts    
# # Example usage:
# root_folder = '/Users/alphatech/Desktop/Educational web app/Data repo'
# exam_name = "CAIA Level 1"
# topic_name = "Hedge Funds"
# chapter_name = "5.2 Macro and Managed Futures Funds"
# subchapter_name = "Systematic Trading"

# pdf_contents = fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name)


# %% [markdown]
# Flashcards generation

# %%
# To help construct our Chat Messages
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

def generate_flashcards(docs, lo, number, difficulty):

    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(docs, embedding=hf)

    retriever = vectorstore.as_retriever()

    # we instantiated the retreiever above
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=chat_model
    )

    # Set logging for the queries
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    context = retriever_from_llm.get_relevant_documents(query=lo)
    
    template = """ You are a knowledgeable assistant capable of generating flashcards using the provided context for reference.
    Given the context delimited by triple backticks, it is your job to generate {number} flashcard(s) with difficulty level {difficulty}.

    Flashcard number:
    Front:
    Back:
    
    Make sure to generate {number} flashcards.
    Ensure the flashcards to be conforming to the context.

            \n```{context}``` """
            
    flash_prompt = ChatPromptTemplate.from_template(template)
    
    user_query = flash_prompt.format_prompt(context=context, number=number, difficulty=difficulty)  # Adjust values accordingly
    flash_response = chat_model(user_query.to_messages())
    
    flash_response = flash_response.content
    # Split the response into individual questions
    flashcards = flash_response.split('\n\n')

    # Remove empty strings from the list
    flashcards_list = [card.strip() for card in flashcards if card.strip()]
    
    return flashcards_list

# number=1
# difficulty="medium"
# cards_list = generate_flashcards(context, number, difficulty)

# %%
from flask import Flask, request, jsonify
import os

import os
import PyPDF2
from docx import Document

from flask_cors import CORS  # Import CORS

app = Flask(__name__)
#run_with_ngrok(app)  # Initialize ngrok when the app is run

CORS(app, supports_credentials=True, allow_headers=["Content-Type"])

def extract_pdf_content(file_path):
    try:      
        loader = Docx2txtLoader(file_path)
        docx_file = loader.load()
        print("Docx file is: ", docx_file)
        return docx_file
    
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_lerningObjective(docx_file):
    document = Document(docx_file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text    
    
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
            
            if pdf_file.endswith('.docx'):  # Check if the file is a docx
                pdf_content = extract_pdf_content(pdf_path)
                
                if pdf_content:
                    pdf_contents.extend(pdf_content)
                    print(f"Successfully extracted content from: {pdf_path}")
                else:
                    print(f"Failed to extract content from: {pdf_path}")
                    
            else:
                print(f"Skipping non-Docx file: {pdf_path}")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_texts = text_splitter.split_documents(pdf_contents)
        print("pdf texts in fetch_pdfs: ", pdf_texts)
    return pdf_texts    


@app.route('/', methods=['GET','POST'])
def fetch_data():
    data = request.json
    
    root_folder = '/Users/alphatech/Desktop/Educational web app/Data repo'
    exam_name = "CAIA Level 1"
    #exam_name = data.get('exam')
    
    chapter_name = data['subChapter']
    print("chapter is: ", chapter_name)
    
    topic_name = data['chapter']
    print("topic is: ", topic_name)
    
    subchapter_name = data['topic']
    print("subchapter is: ", subchapter_name)

    if exam_name is None:
        return jsonify({'error': 'Exam name not provided'}), 400

    docs_text = fetch_pdfs(root_folder, exam_name, topic_name, chapter_name, subchapter_name)
    
    print("splitted text",docs_text)
    
    doc = "Learning Objective.docx"
    lo_path = os.path.join(root_folder, exam_name, topic_name, chapter_name, subchapter_name, doc)
    lo = extract_lerningObjective(lo_path)
    
    difficulty = data['difficulty']
    print(difficulty)
    
    
    if "selectedGenerationOfFlashcards" in data:
        flash_number = data["selectedGenerationOfFlashcards"]
        if flash_number:
            cards_list = generate_flashcards(docs_text, lo, flash_number, difficulty)
            return jsonify(cards_list)

    else:
        question_type = data['typeOfQuestions']
        print(question_type)
        
        number = data['numberOfQuestions']
        print(number)
    
        math_number = data['mathematicsDifficulty']
        print(math_number)   
        
        # Generating theory questions
        questions_response = generate(question_type, docs_text, lo, number, difficulty).content
        
        # Splitting the response into individual questions
        all_questions = []
        all_answers = []
        for question in questions_response.split('\n\n'):
            if question.strip():
                all_questions.append(question.strip())
                all_answers.append(answers_response(question, question_type, docs_text))

        
        # Generating math questions
        doc = "Math.docx"
        file_path = os.path.join(root_folder, exam_name, topic_name, chapter_name, subchapter_name, doc)
        math_response = None
        try:
            math_response = math_generate(question_type, file_path, difficulty=difficulty, math_number=math_number).content
        except Exception as e:
            print(f"Error generating math questions: {e}")

        if math_response:
            # Splitting the response into individual questions
            for math_question in math_response.split('\n\n'):
                if math_question.strip():
                    all_questions.append(math_question.strip())
                    all_answers.append(math_answers(file_path, math_question))
        else:
            print("Skipping math questions due to previous error.")


        return jsonify(all_questions, all_answers)



if __name__ == '__main__':
    app.run(debug=False)



