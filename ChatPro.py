
# %%
import os

# To help construct our Chat Messages
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# We will be using ChatGPT model (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"

# %%
chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# %%
from langchain import PromptTemplate

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

def generate(question_type, docs, lo, number, difficulty):

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

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

# %% [markdown]
# Answers to the questions

# %%
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter
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

        
        vectorstore = FAISS.from_documents(data, embedding=embeddings)
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
        


# %% [markdown]
# Math questions

# %%
from langchain_community.document_loaders import Docx2txtLoader

def math_generate(question_type, file_path, difficulty, math_number):
    if not os.path.exists(file_path):
        print("No Math!")
        return None

    loader = TextLoader(file_path)
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


# %% [markdown]
# Math answers

# %%
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

def math_answers(file_path, math_question):
    if math_question:
        loader = TextLoader(file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        math_data = text_splitter.split_documents(data)

        math_vectorstore = FAISS.from_documents(math_data, embedding=embeddings)
        
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
 


# %% [markdown]
# Flashcards generation

# %%
# To help construct our Chat Messages
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

def generate_flashcards(docs, lo, number, difficulty):

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

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
from langchain_community.document_loaders import TextLoader
import os
from docx import Document

from flask_cors import CORS  # Import CORS

app = Flask(__name__)
#run_with_ngrok(app)  # Initialize ngrok when the app is run

CORS(app, supports_credentials=True, allow_headers=["Content-Type"])

def extract_pdf_content(file_path):
    try:      
        loader = TextLoader(file_path)
        docx_file = loader.load()
        print("tex file is: ", docx_file)
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
            
            if pdf_file.endswith('.tex'):  # Check if the file is a tex
                pdf_content = extract_pdf_content(pdf_path)
                
                if pdf_content:
                    pdf_contents.extend(pdf_content)
                    print(f"Successfully extracted content from: {pdf_path}")
                else:
                    print(f"Failed to extract content from: {pdf_path}")
                    
            else:
                print(f"Skipping non-tex file: {pdf_path}")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_texts = text_splitter.split_documents(pdf_contents)
        #print("tex texts in fetch_pdfs: ", pdf_texts)
    return pdf_texts    


@app.route('/', methods=['GET','POST'])
def fetch_data():
    data = request.json
    
    root_folder = '/Users/alphatech/Desktop/FGW_latexData'
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
    
    #print("splitted text",docs_text)
    
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
        doc = "Math.tex"
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
