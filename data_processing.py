import logging
from datetime import date

from dateutil.relativedelta import relativedelta

from data_generation import generate_dataset
import pandas as pd
import os
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SKLearnVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from Config import Config

# Load OpenAI API key from Streamlit secrets
openai_api_key = Config.openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key


@st.cache_data
@st.cache_data
def load_and_process_data(file_path, chunk_size=1000):
    try:
        logging.info(f"Loading data from {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"No read permission for the file {file_path}")

        processed_data = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                processed_data.append(create_prompt_response(row))
        return processed_data

    except Exception as e:

        return []


def create_prompt_response(entry):
    prompt = (
        f"I'm a {entry['Age']}-year-old {entry['Gender']} looking to invest in {entry['Avenue']} "
        f"for {entry['Purpose']} over the next {entry['Duration']} years. What are my options?"
    )
    response = (
        f"Based on your preferences, here are your investment options:\n"
        f"- Fixed Deposits: {entry['Fixed_Deposits']} offers a rate of {entry['Fixed_Deposits_Rate']:.2f}% for regular customers and {entry['Fixed_Deposits_Senior_Rate']:.2f}% for senior citizens.\n"
        # f"Based on your preferences, here are your investment options:\n"
        f"- Mutual Funds: {entry['Mutual_Funds']}\n"
        f"- Equity Market: {entry['Equity_Market']}\n"
        f"- Debentures: {entry['Debentures']}\n"
        f"- Government Bonds: {entry['Government_Bonds']}\n"
        f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
        f"- PPF: {entry['PPF']}\n"
        f"- Gold: {entry['Gold']}\n"
        f"Factors considered: {entry['Factor']}\n"
        f"Objective: {entry['Objective']}\n"
        f"Expected returns: {entry['Expect']}\n"
        f"Investment monitoring: {entry['Invest_Monitor']}\n"
        f"Reasons for choices:\n"
        f"- Equity: {entry['Reason_Equity']}\n"
        f"- Mutual Funds: {entry['Reason_Mutual']}\n"
        f"- Bonds: {entry['Reason_Bonds']}\n"
        f"- Fixed Deposits: {entry['Reason_FD']}\n"
        f"Source of information: {entry['Source']}\n"
    )
    return {"prompt": prompt, "response": response}


def create_documents(prompt_response_data):
    logging.info(f"Creating {len(prompt_response_data)} documents")
    return [Document(page_content=f"Prompt: {entry['prompt']}\nResponse: {entry['response']}") for entry in
            prompt_response_data]


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    logging.info(f"Splitting {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Created {len(split_docs)} split documents")
    return split_docs

def setup_data_and_vectorstore():
    # Generate sample data
    months = [date.today() - relativedelta(months=+3), date.today() - relativedelta(months=+6)]
    dataset = generate_dataset(50, months)
    df = pd.DataFrame(dataset)
    df['content'] = [f"Based on the following customer data: {data}, suggest suitable banking lending products." for
                     data in dataset]
    documents = [Document(page_content=row["content"], metadata={"class": row["Age"]}) for _, row in df.iterrows()]

    # Load and process the CSV data
    csv_data = load_and_process_data(Config.DATA_FILE)
    csv_documents = create_documents(csv_data)

    # Combine all documents and create the vector store
    all_documents = documents + csv_documents
    texts = split_documents(all_documents)
    return create_vector_db(texts)
@st.cache_resource
def create_vector_db(_texts):
    logging.info("Creating vector database")
    openai_embeddings = OpenAIEmbeddings()
    try:
        vectordb = SKLearnVectorStore.from_documents(
            documents=_texts,
            embedding=openai_embeddings,
            algorithm="brute",
            n_neighbors=5
        )
        return vectordb
    except Exception as e:
        logging.error(f"An error occurred while creating the vector database: {e}")
        st.error(f"An error occurred while creating the vector database: {e}")
        return None


@st.cache_resource
def create_qa_chain(_sklearn_store):
    logging.info("Creating QA chain")
    openai_llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="Based on the following customer data and financial information: {context}, suggest suitable banking lending products and investment strategies in the following format:\n\n1. Product/Strategy 1: Description\n2. Product/Strategy 2: Description\n3. Product/Strategy 3: Description\nProvide detailed recommendations."
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=openai_llm,
        chain_type="stuff",
        retriever=_sklearn_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
