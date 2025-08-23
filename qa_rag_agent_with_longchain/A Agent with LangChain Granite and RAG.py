#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # <a id='toc1_'></a>[Build a grounded Q/A Agent with LangChain, Granite 3 and RAG](#toc0_)
# 

# Estimate time: **30** minutes
# 

# This guided project demonstrates how to use LangChain to create a question and answer agent based on a large language model (LLM) and retrieval augmented generation (RAG) technology.
# 

# **Table of contents**<a id='toc0_'></a>    
# - [Build a grounded Q/A Agent with LangChain, Granite and RAG](#toc1_)    
#   - [Introduction](#toc1_1_)    
#   - [What does this guided project do?](#toc1_2_)    
#   - [Objectives](#toc1_3_)    
#   - [Background](#toc1_4_)    
#     - [What is a large language model (LLM)](#toc1_4_1_)    
#     - [What is LangChain](#toc1_4_2_)    
#     - [What is watsonx Granite](#toc1_4_3_)    
#       - [Benefits of watsonx Granite](#toc1_4_3_1_)    
#     - [What is retrieval augmented generation (RAG)](#toc1_4_4_)    
#     - [What is a question-answering agent](#toc1_4_5_)    
#   - [Setup](#toc1_5_)    
#     - [Installing required libraries](#toc1_5_1_)    
#   - [Watsonx API credentials and project_id](#toc1_6_)    
#   - [Load document data and build knowledge base](#toc1_7_)    
#   - [Create an embedding model](#toc1_8_)    
#   - [Watsonx.ai Embedding with LangChain](#toc1_9_)    
#   - [Foundation models on watsonx.ai](#toc1_10_)    
#     - [Define the model](#toc1_10_1_)    
#     - [Define the model parameters](#toc1_10_2_)    
#     - [LangChain CustomLLM wrapper for watsonx model](#toc1_10_3_)    
#     - [Generate a retrieval augmented response to a question](#toc1_10_4_)    
#     - [Question-answering agent](#toc1_10_5_)    
#   - [Exercises](#toc1_11_)    
#     - [Exercise 1 - Change the query](#toc1_11_1_)    
#     - [Exercise 2 - Change the query](#toc1_11_2_)    
#   - [Authors](#toc1_12_)    
#   - [Contributors](#toc1_13_)    
#   - [Change Log](#toc1_14_)  
# 

# ## <a id='toc1_1_'></a>[Introduction](#toc0_)
# 
# Imagine that you have a large collection of documents and you want to find the answer to a question. You could read through all of the documents to find the answer, but that would be a time-consuming process. Instead, you can use a question-answering agent to find the answer for you. This project explains how to use LangChain to build a question-answering agent that leverages a large language model (LLM) and retrieval augmented generation (RAG) technology to find the answer to a question.
# 

# ## <a id='toc1_2_'></a>[What does this guided project do?](#toc0_)
# 
# Leveraging the IBM watsonx Granite Generation 3 LLM and LangChain, you learn to set up and configure these tools to create a highly accurate RAG pipeline. This hands-on project is perfect for data scientists, AI enthusiasts, and developers who want to acquire practical AI skills that can be applied in real-world scenarios. In just 30 minutes, you gain valuable experience that will enhance your portfolio and open up new possibilities in the field of artificial intelligence.
# 

# ## <a id='toc1_3_'></a>[Objectives](#toc0_)
# 
# After completing this lab you will be able to:
# 
# - Understand how to use the IBM watsonx Granite LLM and its applications in AI-driven solutions.
# - Develop the skills to create a RAG pipeline.
# - Gain practical experience in developing a question-answering agent that can be used in various real-world applications.
# 

# ## <a id='toc1_4_'></a>[Background](#toc0_)
# 
# ### <a id='toc1_4_1_'></a>[What is a large language model (LLM)](#toc0_)
# 
# [Large language models](https://www.ibm.com/topics/large-language-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829) are a type of artificial intelligence model that is trained on a large corpus of text data. LLMs are designed to generate human-like text responses to a wide range of questions. They are based on the Transformer architecture and are pretrained on a variety of language tasks to improve their performance.
# 
# ### <a id='toc1_4_2_'></a>[What is LangChain](#toc0_)
# 
# [LangChain](https://www.ibm.com/topics/langchain?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829) is an open source orchestration framework for the development of applications using LLMs. Available in both Python- and JavaScript-based libraries, LangChain’s tools and APIs simplify the process of building LLM-driven applications like [chatbots](https://www.ibm.com/topics/chatbots?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829) and [virtual agents](https://www.ibm.com/topics/virtual-agent?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829). 
# 
# ### <a id='toc1_4_3_'></a>[What is watsonx Granite](#toc0_)
# 
# [Watsonx Granite](https://www.ibm.com/granite?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829) is a family of AI models that are built for business and engineered from scratch to help ensure trust and scalability in AI-driven applications.
# 
# #### <a id='toc1_4_3_1_'></a>[Benefits of watsonx Granite](#toc0_)
# 
# - **Open**: With a principled approach to data transparency, model alignment, and security red teaming, IBM has been delivering truly open source Granite models under an Apache 2.0 license to empower developers to bring trusted, safe generative AI into mission-critical applications and workflows. 
# - **Performant**: IBM Granite models deliver best-in-class performance in coding, and above-par performance in targeted language tasks and use cases at lower latencies, with continuous, iterative improvements by using pioneering techniques from IBM Research and contributions from open source.
# - **Efficient**: With a fraction of the compute capacity, inferencing costs, and energy consumption demanded by general-purpose models, Granite models enable developers to experiment, build, and scale more generative AI applications while staying well within the budgetary limits of their departments.
# 
# 
# ### <a id='toc1_4_4_'></a>[What is retrieval augmented generation (RAG)](#toc0_)
# 
# LLMs can be inconsistent. Sometimes they correctly answer the questions, and other times they respond with random facts from their training data. If they occasionally sound like they have no idea what they’re saying, it’s because they don’t. LLMs know how words relate statistically, but not what they mean.
# 
# [RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM’s internal representation of information.
# 
# ### <a id='toc1_4_5_'></a>[What is a question-answering agent](#toc0_)
# 
# A [question-answering agent](https://huggingface.co/tasks/question-answering) is a type of artificial intelligence system that is designed to generate responses to questions. It is typically based on an LLM that is trained on a variety of language tasks. Question-answering agents are used in a wide range of applications, including chatbots, search engines, and virtual assistants.
# 

# ## <a id='toc1_5_'></a>[Setup](#toc0_)
# 
# For this lab, you use the following libraries:
# 
# *   [`langchain`](https://pypi.org/project/langchain/) for integrating language models and retrieval models.
# *   [`ibm-watsonx-ai`](https://pypi.org/project/ibm-watsonx-ai/) for accessing the watsonx Granite language model.
# *   [`wget`](https://pypi.org/project/wget/) for downloading files from the internet.
# *   [`sentence-transformers`](https://pypi.org/project/sentence-transformers/) for computing dense vector representations for sentences, paragraphs, and images.
# *   [`chromadb`](https://pypi.org/project/chromadb/) for an open source embedding database.
# *   [`pydantic`](https://pypi.org/project/pydantic/) for data validation.
# *   [`sqlalchemy`](https://pypi.org/project/sqlalchemy/) for SQL toolkit and Object-Relational Mapping (ORM).
# 

# ### <a id='toc1_5_1_'></a>[Installing required libraries](#toc0_)
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You must run the following cell__ to install them. Please wait until it completes.
# 
# This step could take **approximately 10 minutes**, so please be patient.
# 
# **NOTE**: If you encounter any issues, please restart the kernel and run the cell again.  You can do that by clicking the **Restart the kernel** icon.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/crvBKBOkg9aBzXZiwGEXbw/Restarting-the-Kernel.png" width="50%" alt="Restart kernel">
# 

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install langchain==0.2.6 | tail -n 1\n!pip install langchain-community==0.2.6 | tail -n 1\n!pip install ibm-watsonx-ai==1.0.10 | tail -n 1\n!pip install langchain_ibm==0.1.8 | tail -n 1\n!pip install wget==3.2 | tail -n 1\n!pip install sentence-transformers==3.0.1 | tail -n 1\n!pip install chromadb==0.5.3 | tail -n 1\n!pip install pydantic==2.8.0 | tail -n 1\n!pip install sqlalchemy==2.0.30 | tail -n 1\n')


# ## <a id='toc1_6_'></a>[Watsonx API credentials and project_id](#toc0_)
# 
# This section provides you with the necessary credentials to access the watsonx API.
# 

# In[1]:


from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
import os


credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )

client = APIClient(credentials)

project_id  = "skills-network"


# ## <a id='toc1_7_'></a>[Load document data and build knowledge base](#toc0_)
# 
# In this section, you load the document data and build the knowledge base that is used to answer questions.
# 
# The most important part of the knowledge base is the document data. Retrieval models are used to retrieve relevant information from the document data. The document data can be stored in a variety of formats, such as text files, PDFs, or databases.
# 
# RAG creates dense embeddings for the document data and stores them in a database. 
# 
# In this project, you use a file and split it into chunks, then embed each chunk using an embedding model, and store the embeddings in a database.
# 
# **CharacterTextSplitter**
# 
# The CharacterTextSplitter is a utility that is used to split documents into smaller chunks based on a specified strategy. In this case, it splits the document into chunks of a defined size without overlapping content between chunks.
# 
# - **Chunk size**: The chunk_size parameter specifies the maximum number of characters in each chunk. 
# - **Chunk overlap**: The chunk_overlap parameter determines the number of characters that should overlap between consecutive chunks.
# 

# In[2]:


import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Define filename and URL
filename = 'state_of_the_union.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/zNYlnZMW6K-9GP72DDizOQ/state-of-the-union.txt'

# Download the file if it does not exist
if not os.path.isfile(filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Load the document
loader = TextLoader(filename)
documents = loader.load()

# Split the document into chunks using CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(texts)}")


# ## <a id='toc1_8_'></a>[Create an embedding model](#toc0_)
# 
# This section creates an embedding model. You use the `ibm-watsonx-ai` library to access the **WatsonX Granite language model.** 
# 

# In[3]:


from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain.vectorstores import Chroma

get_embedding_model_specs(credentials.get('url'))

# Part 1: Create Embedding Model
# Set up the WatsonxEmbeddings object
embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    project_id=project_id
    )

# Part 2: Embed Documents and Store
docsearch = Chroma.from_documents(texts, embeddings)

# Let us print several embedding vectors.
# Generate and print embedding vectors for a sample of the documents
sample_texts = texts[:3]  # Taking a sample of 3 documents for demonstration
sample_embeddings = embeddings.embed_documents([doc.page_content for doc in sample_texts])

print("Sample Embedding Vectors:")
for i, embedding in enumerate(sample_embeddings):
    print(f"Document {i + 1} Embedding Vector: Length: {len(embedding)}; {embedding}")


# ## <a id='toc1_9_'></a>[Watsonx.ai Embedding with LangChain](#toc0_)
# 
# You can use the help function to get detailed information about the **WatsonxEmbeddings** class, including its methods, attributes, and usage. This is useful for understanding how to properly utilize the class in your project.
# 

# In[4]:


help(WatsonxEmbeddings)


# ## <a id='toc1_10_'></a>[Foundation models on watsonx.ai](#toc0_)
# 
# IBM WatsonX provides various foundation models that can be used for different tasks. 
# 
# This section shows how to use the [Granite model](https://www.ibm.com/new/ibm-granite-3-0-open-state-of-the-art-enterprise-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+grounded+Q%2FA+Agent+with+LangChain%2C+Granite+and+RAG-v1_1729612730) using [LangChain](https://python.langchain.com/v0.2/docs/introduction/).
# 
# ### <a id='toc1_10_1_'></a>[Define the model](#toc0_)
# 
# In this project, you use `ibm/granite-3-8b-instruct`.
# 
# The [Granite Gen-3 8 Billion instruct model (granite-3-8b-instruct)](https://www.ibm.com/new/ibm-granite-3-0-open-state-of-the-art-enterprise-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+grounded+Q%2FA+Agent+with+LangChain%2C+Granite+and+RAG-v1_1729612730) is a new, instruction-tuned, dense decoder-only LLM. Trained using a novel two-phase method on over 12 trillion tokens of carefully vetted data across 12 different natural languages and 116 different programming languages, the developer-friendly Granite 3.0 8B Instruct is a workhorse enterprise model intended to serve as a primary building block for sophisticated workflows and tool-based use cases. Granite 3.0 8B Instruct matches leading similarly-sized open models on academic benchmarks while outperforming those peers on benchmarks for enterprise tasks and safety.
# 
#  Previous generations of Granite models prioritized specialized use cases, excelling at domain-specific tasks across a diverse array of industries including finance, legal, code and academia. In addition to offering even greater efficacy in those arenas, IBM Granite 3.0 models match—and, in some cases, exceed—the general performance of leading open weight LLMs across both academic and enterprise benchmarks.
# 

# ### <a id='toc1_10_2_'></a>[Define the model parameters](#toc0_)
# 

# In[5]:


from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["\n"],
}


# ### <a id='toc1_10_3_'></a>[LangChain CustomLLM wrapper for WatsonX model](#toc0_)
# 
# This section initializes the `WatsonxLLM` from `LangChain` with the model ID and model parameters.
# 

# In[6]:


from langchain_ibm import WatsonxLLM


# Create a dictionary to store credential information
credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

# Indicate the model we would like to initialize. In this case, ibm/granite-3-8b-instruct.
model_id    = 'ibm/granite-3-8b-instruct'

# Initialize some watsonx.ai model parameters
params = {
        "decoding_method": "greedy",
        "temperature": 0.4, 
        "min_new_tokens": 1,
        "max_new_tokens": 100,
        #"stop_sequences":["\n"]
    }
project_id  = "skills-network" # <--- NOTE: specify "skills-network" as your project_id
# space_id    = None
# verify      = False

watsonx_granite = WatsonxLLM(
    model_id=model_id, 
    url=credentials["url"], 
    params=params, 
    project_id=project_id, 
)


# ### <a id='toc1_10_4_'></a>[Generate a retrieval augmented response to a question](#toc0_)
# 
# This section builds the question answering chain to automate the RAG pipeline.
# 

# In[7]:


from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())


# ### <a id='toc1_10_5_'></a>[Question-answering agent](#toc0_)
# 
# This section answers the questions by using the RAG pipeline.
# 

# In[8]:


query = "What did the president say about highway and bridges in disrepair"
qa.invoke(query)

# in text, it is mentioned: this year we will start fixing over 65,000 miles of highway and 1,500 bridges in disrepair. 


# ## <a id='toc1_11_'></a>[Exercises](#toc0_)
# 
# ### <a id='toc1_11_1_'></a>[Exercise 1 - Change the query](#toc0_)
# 
# Please check the answer to the following question: "What did the president say about the infrastructure rank in the world?"
# 

# In[9]:


## your solution here
query = "What did the president say about the infrastructure rank in the world?"
qa.invoke(query)


# <details>
#     <summary>Click here for solution</summary>
# 
# ```python
# query = "What did the president say about the infrastructure rank in the world?"
# qa.invoke(query)
# 
# # in text, it is mentioned: Now our infrastructure is ranked 13th in the world. 
# ```
# ```
# {'query': 'What did the president say about the infrastructure rank in the world?',
# 
# 'result': ' The president mentioned that America used to have the best roads, bridges, and airports on Earth, but now our infrastructure is ranked 13th in the world.'}
# ```
# 
# </details>
# 

# ### <a id='toc1_11_2_'></a>[Exercise 2 - Change the query](#toc0_)
# 
# Please check the answer to the following question: "What did the president say about a unity agenda for the nation? What is the first thing we can do together?"
# 

# In[10]:


## your solution here
query = query = "What did the president say about a Unity Agenda for the Nation? What is the first thing we can do together?"
qa.invoke(query)


# <details>
#     <summary>Click here for solution</summary>
# 
# ```python
# query = query = "What did the president say about a Unity Agenda for the Nation? What is the first thing we can do together?"
# qa.invoke(query)
# 
# # in text, it is mentioned: So tonight I'm offering a Unity Agenda for the Nation. Four big things we can do together. First, beat the opioid epidemic. 
# ```
# ```
# {'query': 'What did the president say about a Unity Agenda for the Nation? What is the first thing we can do together?',
#  'result': ' The president mentioned a Unity Agenda for the Nation, in which he proposed four big things that the country can do together. The first thing on this list is beating the opioid epidemic.\n'}
# ```
# 
# </details>
# 

# ## <a id='toc1_12_'></a>[Authors](#toc0_)
# 
# 
# [Ricky Shi](https://www.linkedin.com/in/ricky-shi-ca/)
# 

# ## <a id='toc1_13_'></a>[Contributors](#toc0_)
# 
# [Faranak Heidari](https://www.linkedin.com/in/faranakhdr)
# 
# [Kang Wang](https://author.skills.network/instructors/kang_wang)
# 
# [Lucy Xu](https://author.skills.network/instructors/lucy_xu)
# 

# ## <a id='toc1_14_'></a>[Change Log](#toc0_)
# 
# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2024-07-03|0.1|Ricky Shi|Create project|
# |2024-10-22|0.2|Faranak Heidari|Update with Gen3 Granite|
# 

# Copyright © 2024 IBM Corporation. All rights reserved.
# 
