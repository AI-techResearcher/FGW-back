# %%
# from torch import cuda, bfloat16
# import transformers

# model_id = 'meta-llama/Llama-2-13b-chat-hf'

# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# # set quantization configuration to load large model with less GPU memory
# # this requires the `bitsandbytes` library
# bnb_config = transformers.BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=bfloat16
# )

# # begin initializing HF items, need auth token for these
# hf_auth = 'HF_AUTH_TOKEN'
# model_config = transformers.AutoConfig.from_pretrained(
#     model_id,
#     use_auth_token=hf_auth
# )

# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     config=model_config,
#     quantization_config=bnb_config,
#     device_map='auto',
#     use_auth_token=hf_auth
# )
# model.eval()
# print(f"Model loaded on {device}")

# %% [markdown]
# The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Llama 2 13B models were trained using the Llama 2 13B tokenizer, which we initialize like so:

# %%
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_id,
#     use_auth_token=hf_auth
# )

# %% [markdown]
# Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.

# %%
# generate_text = transformers.pipeline(
#     model=model, tokenizer=tokenizer,
#     return_full_text=True,  # langchain expects the full text
#     task='text-generation',
#     # we pass model parameters here too
#     temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
#     max_new_tokens=512,  # mex number of tokens to generate in the output
#     repetition_penalty=1.1  # without this output begins repeating
# )

# %%
# from langchain.llms import HuggingFacePipeline

# llm = HuggingFacePipeline(pipeline=generate_text)

# %%
# from langchain.chains import RetrievalQA

# rag_pipeline = RetrievalQA.from_chain_type(
#     llm=llm, chain_type='stuff',
#     retriever=vectorstore.as_retriever()
# )

# %%
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# We will be using ChatGPT model (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI

# To parse outputs and get structured data back
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader
from flask_cors import CORS
from flask import Flask, request, jsonify


def extract_text_from_pdf(file_path):
    
    try:
        loader = Docx2txtLoader(file_path)
        docx_file = loader.load()
        print("Docx file is: ", docx_file)
        return docx_file
    
    except Exception as e:
        print(f"Error reading PDF file '{file_path}': {e}")
        return " "



def extract_text_from_all_pdfs(root_folder):
    docs = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.docx'):
                file_path = os.path.join(root, file)
                data = extract_text_from_pdf(file_path)
                docs.extend(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pdf_texts = text_splitter.split_documents(docs)
    return pdf_texts

# %%
os.environ["OPENAI_API_KEY"] = "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"

chat_model = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# %%
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        
        root_folder = '/Users/alphatech/Desktop/Educational web app/fgwpro-main2'
        exam_name = "CAIA Level 1"
        root_folder = os.path.join(root_folder, exam_name)
        
        pdf_texts = extract_text_from_all_pdfs(root_folder)
        
        vectorstore = FAISS.from_documents(pdf_texts, embedding=embeddings)
        
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=chat_model, chain_type='stuff',
            retriever=vectorstore.as_retriever()
        )
        
        question = data.get('question')
        if question is None or question.strip() == '':
                raise ValueError("Invalid or empty question")
            
        result = rag_pipeline(question)

        response = {'query': question, 'result': result}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port = 5002)


