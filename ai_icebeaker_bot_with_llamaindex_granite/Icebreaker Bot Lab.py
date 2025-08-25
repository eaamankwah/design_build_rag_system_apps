#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Build an AI Icebreaker Bot with LlamaIndex & IBM Granite**
# 

# Estimated time needed: **40** minutes
# 

# ## <a id='toc1_0_'></a>[Overview](#toc0_)
# 

# Imagine you’re heading to a big networking event, surrounded by potential employers and industry leaders. You want to make a great first impression, but you’re struggling to come up with more than the usual, "What do you do?"
# 
# Now, picture having an AI-powered tool that does the research for you. You input a name, and within seconds, the bot, powered by **LlamaIndex** and **IBM watsonx**, searches LinkedIn, generating personalized icebreakers based on someone’s career highlights, interests, and even fun facts. Instead of generic questions, you start with something unique and meaningful.
# 
# The AI icebreaker bot uses **natural language processing (NLP)** to create tailored conversation starters that help you stand out. By the end of this project, you’ll have built a tool that helps make introductions smoother, more personal, and more memorable—perfect for networking, job interviews, or any social setting.
# 

# ## <a id='toc0_'></a>[Table of contents](#toc0_) 
#    
#   - [Background](#toc2_0_)    
#     - [What are Large Language Models (LLMs)?](#toc2_0_1_)    
#     - [What is IBM watsonx?](#toc2_0_2_)    
#     - [What is LlamaIndex?](#toc2_0_3_)    
#     - [What is RAG?](#toc2_0_4_)
#       - [What are RAG stages](#toc2_0_4_1_)
#     - [What is prompt engineering?](#toc2_0_5_)
#   - [Objectives](#toc3_0_)     
#   - [Setup](#toc4_0_)
#     - [Installing required libraries](#toc4_0_1_)     
#     - [Importing required libraries](#toc4_0_2_)    
#   - [The whole process](#toc5_0_)    
#     - [Extracting LinkedIn profile data](#toc5_0_1_)
#         - [What is ProxyCurl?](#toc5_0_1_2_)
#         - [Signing up for ProxyCurl](#toc5_0_1_3_)
#         - [Getting and replacing your API key](#toc5_0_1_4_)
#         - [Using mock data](#toc5_0_1_5_)     
#     - [Splitting LinkedIn profile data into nodes](#toc5_0_2_)
#     - [Indexing and storing LinkedIn profile data](#toc5_0_3_)
#     - [Querying the indexed data](#toc5_0_4_)
#     - [Putting everything together: Building a chatbot interface](#toc5_0_5_)      
#   - [Exercises](#toc6_0_)    
#     - [Exercise 1 - Try with another LLM](#toc6_0_1_)    
#     - [Exercise 2 - Work with another LinkedIn profile URL](#toc6_0_2_)    
# 

# ## <a id='toc2_0_'></a>[Background](#toc0_)
# 
# ### <a id='toc2_0_1_'></a>[What are Large Language Models (LLM)?](#toc0_)
# 
# [Large language models](https://www.ibm.com/topics/large-language-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829) are a type of artificial intelligence (AI) models that are trained on a large corpus of text data. LLMs are designed to generate human-like text responses to a wide range of questions. They are based on the Transformer architecture and are pretrained on a variety of language tasks to improve their performance.
# 
# ### <a id='toc2_0_2_'></a>[What is IBM watsonx?](#toc0_)
# 
# [IBM watsonx](https://www.ibm.com/watsonx?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Learn+New+Fields+Fast+with+Watsonx+and+LangChain_v1_1721853863) is a suite of AI tools and services that are designed to help developers build and deploy AI-driven applications. watsonx provides a range of APIs and tools that make it easy to integrate AI capabilities into applications, including natural language processing, computer vision, and speech recognition.
# 
# ### <a id='toc2_0_3_'></a>[What is LlamaIndex?](#toc0_)
# 
# [LlamaIndex](https://www.ibm.com/topics/llamaindex?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Icebreaker+Bot+Lab_v1_1726516829) is an open-source data orchestration platform for creating large language model (LLM) applications. LlamaIndex is accessible in Python and TypeScript, and it uses a set of tools and features to ease the context augmentation process for [generative AI (gen AI)](https://www.ibm.com/topics/generative-ai?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Icebreaker+Bot+Lab_v1_1726516829) use cases via a Retrieval-Augmented (RAG) pipeline.
# 
# ### <a id='toc2_0_4_'></a>[What is RAG?](#toc0_)
# 
# While LLMs are built using extensive datasets, they don't naturally include your specific data. [Retrieval-Augmented Generation (RAG)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Icebreaker+Bot+Lab_v1_1726516829) solves this issue by merging your data with the existing knowledge of LLMs. Throughout this guide, you'll frequently see mentions of RAG, as it's a critical technique used in query and chat engines, as well as agents, to boost their performance.
# 
# In a RAG setup, your data is first loaded, processed, and indexed for quick retrieval. When a user submits a query, the system searches the index to find the most relevant information from your data. This contextual information is then combined with the user's query and sent to the LLM, which generates a response based on this refined data.
# 
# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ISxumwmkqbJnsY9KN9PzAw/Rag%20Basics.png" width="80%" alt="ragbascis">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 
# ### <a id='toc2_0_4_1_'></a>[What are RAG stages?](#toc0_)
# 
# In the RAG framework, there are five key stages, though this lab will concentrate on the first four. These stages are essential for most large-scale applications you may develop. They include:
# 
# - **Loading**: This step involves importing your data into the workflow, regardless of the source—such as text files, PDFs, websites, databases, or APIs. LlamaHub provides various connectors to simplify this process.
# 
# - **Indexing**: In this phase, a data structure is built for efficient querying. For LLMs, this often means creating vector embeddings, numerical representations that capture the meaning of the data, along with metadata strategies to ensure precise and context-aware data retrieval.
# 
# - **Storing**: Once indexing is complete, it's important to save the index and its associated metadata to avoid re-indexing the data later.
# 
# - **Querying**: Based on the indexing strategy, there are various ways to perform queries using LLMs and LlamaIndex structures. This can include sub-queries, multi-step queries, or hybrid techniques.
# 
# - **Evaluation**: A crucial step in any workflow is evaluating the effectiveness of your approach. Evaluation provides objective metrics to measure the accuracy, relevance, and speed of query results when comparing different methods or making adjustments.
# 
# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/0zwxlxbQ17ID4Nn2iFjn_Q/Rag%20Stages.png" width="70%" alt="ragstages">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 
# 
# ### <a id='toc2_0_5_'></a>[What is prompt engineering?](#toc0_)
# 
# [Prompt Engineering](https://www.ibm.com/topics/prompt-engineering?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Icebreaker+Bot+Lab_v1_1727202669) is the process of designing and refining the inputs (prompts) given to language models such as GPT to get desired outputs. It involves crafting questions, instructions, or context in a way that guides the model to generate accurate, relevant, or creative responses. Good prompt engineering can improve the quality, specificity, and usefulness of the generated text, making it a critical skill for leveraging large language models in various applications like chatbots, content generation, or data analysis.
# 

# ## <a id='toc3_0_'></a>[Objectives](#toc0_)
# 
# After completing this lab, you will be able to:
# 
# - Understand how to use **LlamaIndex** and **IBM watsonx** for personalized information retrieval.
# - Learn how to search and extract relevant data from **LinkedIn** for icebreaker generation.
# - Develop skills in using **AI and natural language processing (NLP)** to generate customized conversation starters based on a person’s online presence.
# - Gain practical experience in applying **AI-powered tools** to automate social interactions, making introductions more engaging and memorable.
# 

# ----
# 

# ## <a id='toc4_0_'></a>[Setup](#toc0_)
# 
# For this lab, you use the following libraries:
# 
# *   [`ibm-watsonx-ai`](https://pypi.org/project/ibm-watsonx-ai/) for accessing the watsonx Granite language model.
# *   [`llama-index-llms-ibm`](https://docs.llamaindex.ai/en/stable/examples/llm/ibm_watsonx/) for communicating with watsonx.ai models using the LlamaIndex and watsonx.ai's LLMs API.
# *   [`llama-index-embeddings-ibm`](https://docs.llamaindex.ai/en/stable/examples/embeddings/ibm_watsonx/) for using watsonx.ai's embedding models.
# *   [`llama-index`](https://www.llamaindex.ai/) for using LlamaIndex framework relevant features.
# 

# ### <a id='toc4_0_1_'></a>[Installing required libraries](#toc0_)
# 
# The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:
# 
# **Note:** We are pinning the version here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.
# 
# This might take approximately 1-2 minutes. Please be patient.
# 
# As we use `%%capture` to capture the installation, you won't see the output process. But after the installation completes, you will see a number beside the cell.
# 

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install ibm-watsonx-ai==1.1.2\n!pip install --user llama-index==0.11.8\n!pip install llama-index-core==0.11.8\n!pip install llama-index-llms-ibm==0.2.0\n!pip install llama-index-embeddings-ibm==0.2.0\n!pip install llama-index-readers-web==0.2.2\n!pip install llama-hub==0.0.79.post1\n!pip install requests==2.32.2\n')


# After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/build-a-hotdog-not-hotdog-classifier-guided-project/images/Restarting_the_Kernel.png" width="50%" alt="Restart kernel">
# 

# ### <a id='toc4_0_2_'></a>[Importing required libraries](#toc0_)
# 
# _We recommend you import all required libraries in one place (here):_
# 

# In[1]:


import os
import time
import json
import requests
import logging
import sys
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.getLogger("ibm_watsonx_ai").setLevel(logging.ERROR)

# IBM Watsonx API Client and Credentials handling
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Core components for handling documents, embeddings, and indices
from llama_index.core import Document, VectorStoreIndex, PromptTemplate, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# LlamaIndex IBM-specific components for LLMs and embeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.embeddings.ibm import WatsonxEmbeddings

# For displaying rich content in Jupyter notebooks (Markdown, etc.)
from IPython.display import display, Markdown

# Disable warnings for a cleaner notebook or console experience
def warn(*args, **kwargs):
    pass
warnings.warn = warn


# ## <a id='toc5_0_'></a>[The whole process](#toc0_)
# 

# ### <a id='toc5_0_1_'></a>[Extracting LinkedIn profile data](#toc0_)
# 

# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BAA7t8vocItB4evA7wuXaA/RAG-Loading.png" width="70%" alt="ragloading">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 
# In a traditional RAG workflow, the first stage is the **Loading** phase, where we typically load external documents to gather knowledge. However, instead of loading pre-existing documents, this project involves reading through a LinkedIn profile for the necessary data. This data will be stored in a vector database, allowing us to retrieve information and answer questions based on the extracted LinkedIn profile.
# 

# >**NOTE**: In this lab, we will demonstrate two methods for obtaining LinkedIn profile data: one using the **ProxyCurl API** and the other **using our pre-generated mock data**. Feel free to choose the option that suits you best.
# 

# #### <a id='toc5_0_1_2_'></a>[What is  ProxyCurl?](#toc0_)
# 
# ProxyCurl is a robust API that allows developers to extract information from various websites, including social media platforms like LinkedIn.
# 
# While LlamaIndex provides a built-in **Web Page Reader** for reading websites, it cannot extract LinkedIn data. To overcome this, we utilize the ProxyCurl API, which provides a reliable and efficient way to extract LinkedIn profiles data.
# 

# #### <a id='toc5_0_1_3_'></a>[Signing up for ProxyCurl](#toc0_)
# 
# To sign up for an account on ProxyCurl and to obtain an API key:
# 
# 1. Go to ProxyCurl's website: [ProxyCurl](https://nubela.co/proxycurl/)
# 2. Create an account: Click on **Sign Up** and create an account by filling in the required information.
# 3. Choose your email type:
#    1. Personal email: You'll receive 10 free credits upon signing up.
#    2. Work email: If you use a work email, you'll receive 100 free credits upon sign-up.
# 
#   
# **NOTE**: For this project, 10 credits are enough. **Each API call costs 2 credits**, so this allows for extracting data from 5 profiles. If you need more credits to work with additional profiles, you can purchase them directly from ProxyCurl, but **this is at your own discretion and responsibility.**
# 

# #### <a id='toc5_0_1_4_'></a>[Getting and replacing your API key](#toc0_)
# 
# Once you've signed up and received your free credits, you will be provided with an API key. This key is necessary to authenticate your requests to ProxyCurl. After obtaining the key, place it in the code block below.
# 

# In[2]:


PROXYCURL_API_KEY = "# Replace with your API Key" 


# Now, you are ready to use ProxyCurl API in your project!
# 

# The code connects to ProxyCurl's LinkedIn API through an API endpoint, defined as `api_endpoint`. The request headers include the necessary authentication using an API key. It's important to replace the placeholder `"PROXYCURL_API_KEY"` with your actual API key for the authentication to work properly. However, if you do not want to use the API, you can opt to load mock data from a pre-made LinkedIn profile JSON file by setting the `mock` parameter to `True`.
# 
# **Parameters and API request:**
# 
# The parameters for the API request are set in `params`. The key parameter is the **LinkedIn profile URL** that you want to scrape, specified as `url`. To manage API usage efficiently, the options `fallback_to_cache` and `use_cache` are used to pull cached data where possible, reducing the likelihood of hitting rate limits or making redundant API calls. Additional data, such as inferred salary, skills, or personal contact information (if available), can also be retrieved.
# 

# #### <a id='toc5_0_1_5_'></a>[Using mock data](#toc0_)
# 

# However, if you don’t want to use the ProxyCurl API, we have provided a pre-made LinkedIn Profile JSON file for you to use. **You can use this mock data by passing `mock=True` to the function below.**
# 

# In[3]:


def extract_linkedin_profile(linkedin_profile_url: str, PROXYCURL_API_KEY: str = None, mock: bool = False) -> dict:
    """Extract LinkedIn profile data using Proxycurl API or loads a premade JSON file if mock is True."""

    start_time = time.time()
    
    if mock:
        print("Using mock data from a premade JSON file...")
        linkedin_profile_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"
        response = requests.get(linkedin_profile_url, timeout=30)
    else:
        # Ensure API key is provided when mock is False
        if not PROXYCURL_API_KEY:
            raise ValueError("PROXYCURL_API_KEY is required when mock is set to False.")
        
        print("Starting to extract the LinkedIn profile...")

        # Set up the API endpoint and headers
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        headers = {
            "Authorization": PROXYCURL_API_KEY
        }

        # Prepare parameters for the request
        params = {
            "url": linkedin_profile_url,
            "fallback_to_cache": "on-error",
            "use_cache": "if-present",
            "skills": "include",
            "inferred_salary": "include",
            "personal_email": "include",
            "personal_contact_number": "include"
        }

        print(f"Sending API request to Proxycurl at {time.time() - start_time:.2f} seconds...")

        # Send API request
        response = requests.get(api_endpoint, headers=headers, params=params, timeout=10)
    
    print(f"Received response at {time.time() - start_time:.2f} seconds...")

    # Check if response is successful
    if response.status_code == 200:
        # Clean the data, remove empty values and unwanted fields
        data = response.json()
        data = {
            k: v
            for k, v in data.items()
            if v not in ([], "", None) and k not in ["people_also_viewed", "certifications"]
        }

        if data.get("groups"):
            for group_dict in data.get("groups"):
                group_dict.pop("profile_pic_url")

        return data
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return {}


# >**NOTE**: We will test out our function below. **Please only run codes in the section of your prefered method.**
# 

# #### Extracting LinkedIn data with ProxyCurl API (Optional)
# 

# In[4]:


profile_url = "https://www.linkedin.com/in/leonkatsnelson/"


# In[5]:


profile_data = extract_linkedin_profile(linkedin_profile_url=profile_url, PROXYCURL_API_KEY=PROXYCURL_API_KEY, mock=False)


# In[6]:


profile_data


# #### Using a pre-made LinkedIn profile JSON file
# 

# In[7]:


profile_data = extract_linkedin_profile(linkedin_profile_url="dummy_url", mock=True)


# In[8]:


profile_data


# ### <a id='toc5_0_2_'></a>[Splitting LinkedIn profile data into nodes](#toc0_)
# 

# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hfC81XkfbupgVpjT-l0Ybg/RAG-Splitting.png" width="70%" alt="ragsplitting">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 
# In the traditional RAG framework, after the Loading stage, the next step is **Splitting**. This involves dividing the data into smaller, manageable chunks that can be efficiently embedded into a vector database and later retrieved for answering user queries.
# 

# #### What is splitting in the RAG process?
# 

# In our project, we need to split the LinkedIn profile data (scraped from ProxyCurl) into smaller segments or nodes. These nodes represent logical chunks of information that can later be queried. 
# 
# Since the LinkedIn profile data is a JSON file, we first convert it into a textual format, and then split the text into smaller nodes of approximately 500 characters each.
# 
# This splitting process ensures that our data is indexed in manageable pieces, making it easier for the model to retrieve relevant information based on user queries.
# 

# In[9]:


def split_profile_data(profile_data):
    """Splits the LinkedIn profile JSON data into nodes."""
    try:
        # The extracted LinkedIn profile data is returned in JSON format. To work with this data more easily, 
        # we first convert it into a text string using the json.dumps() function. 
        # This transformation allows us to manipulate the data in subsequent steps, 
        # such as splitting it for further processing.
        profile_json = json.dumps(profile_data)

        # Once the JSON string is created, it is wrapped inside a `Document` object. 
        # This step is necessary because the `Document` format is required for the splitting 
        # and processing steps that follow. The `Document` serves as a container for the profile data, 
        # enabling structured handling of the information.
        document = Document(text=profile_json)

        # To break down the document into smaller parts, we utilize the `SentenceSplitter` class. 
        # This tool splits the document into manageable chunks, called `nodes`. 
        # The parameter `chunk_size=500` is used, meaning each node will contain approximately 500 characters. 
        # This ensures that each chunk is small enough for efficient processing while maintaining coherence for the model to understand.
        splitter = SentenceSplitter(chunk_size=500)

        # Once the document is split, the function returns a list of nodes. 
        # Each node represents a portion of the original LinkedIn profile data, 
        # and these chunks will later be stored in a vector database. 
        # This step is crucial for enabling efficient indexing and retrieval in future operations.
        nodes = splitter.get_nodes_from_documents([document])
        return nodes
        
    # The entire function is wrapped in a `try-except` block to manage potential errors. 
    # If something goes wrong during the process, the function catches the error, 
    # prints an error message for debugging, and returns an empty list. 
    # This helps ensure the program remains stable, even when issues arise.
    except Exception as e:
        print(f"Error in split_profile_data: {e}")
        return []


# >For the full documentation of `Document/Nodes` in LlamaIndex, you can refer [here](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/). An example usage to create the `Document` object is as below.
# 
# ```python
# from llama_index.core import SimpleDirectoryReader
# 
# documents = SimpleDirectoryReader("./data").load_data()
# 

# >For the full documentation of `SentenceSplitter` in LlamaIndex, you can refer [here](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/#llama_index.core.node_parser.SentenceSplitter). An example usage of `SentenceSplitter` is as below.
# 
# ```python
# splitter = SentenceSplitter(chunk_size=500)
# nodes = splitter.get_nodes_from_documents(documents)
# 

# We will test the function with the profile data we scraped previously.
# 

# In[10]:


nodes = split_profile_data(profile_data)

print(f"Number of nodes created: {len(nodes)}")

# Print the first few nodes for inspection
for i, node in enumerate(nodes[:5]):
    print(f"\nNode {i+1}:")
    print(node.get_text())


# As you can see, the JSON data has been split into 17 nodes, ready to be embedded in a vector database.
# 

# ### <a id='toc5_0_3_'></a>[Indexing and storing LinkedIn profile data](#toc0_)
# 

# In this step, we move to **Indexing** and **Storing** these nodes for efficient retrieval. This step is critical for building a foundation for RAG, where we retrieve relevant data to answer user queries.
# 
# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/AEEhG4hgyHo0s69ZmGd9hQ/RAG-Index-Storing.png" width="70%" alt="rag-index-store">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 

# #### Indexing in the RAG workflow
# 

# An `Index` is a key data structure in **LlamaIndex** that allows us to retrieve the relevant context in response to a user query. Indexing is vital because it enables quick and efficient access to relevant chunks of the data, which makes question-answering both faster and more accurate.
# 
# At a high level, `Indexes` are built from the `Nodes` created in the previous splitting stage. These indexes are then used to construct `Retrievers` and `Query Engines`, which power question-answering interactions and conversational experiences based on your data.
# 

# #### What is vector embedding?
# 

# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/FL6t-jfoJfelsfQdshoB4Q/Vector%20Embedding-1.png" width="70%" alt="vector-embedding">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing//">source</a></figcaption>
# </figure>
# 
# Before we can index the nodes, we need to convert them into vector representations. This process, known as `embedding`, allows the model to better understand the text data and compare its semantic similarity to user queries. Vectors enable efficient search and retrieval operations, as similar pieces of text will have similar vector representations.
# 
# To embed the nodes, we’ll use an `IBM watsonx Embedding model`. Once the data is embedded, we store these vectors in a vector database for retrieval during query processing.
# 

# #### Setting up the watsonx embedding model
# 

# To convert the document nodes into vector representations, we need to use an embedding model. In this project, we’ll be calling an embedding model `ibm/slate-125m-english-rtrvr` from IBM watsonx.ai.
# 

# In[11]:


def create_watsonx_embedding():
    """Creates an IBM Watsonx Embedding model for vector representation."""
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        truncate_input_tokens=3,
    )
    return watsonx_embedding


# #### Creating the vector database with `VectorStoreIndex`
# 

# Now that we have our embedding model ready, we can create a `vector database` where the document nodes will be stored as vectors. This will allow us to quickly retrieve relevant information based on user queries.
# 

# In[12]:


def vector_database(nodes):
    """Stores the document chunks (nodes) in a vector database."""
    try:
        # We first call the `create_watsonx_embedding()` function to 
        # get the IBM watsonx embedding model, which will embed our nodes.
        embedding_model = create_watsonx_embedding()

        # The VectorStoreIndex class is used to embed the nodes and 
        # store the resulting vector representations in a vector database.
        index = VectorStoreIndex(
            nodes=nodes, # These are the chunks of text (or nodes) that were created in the previous splitting step.
            embed_model=embedding_model, # The embedding model used to convert text into vectors.
            show_progress=False # This hides the progress bar during the embedding process, 
                                # but you can set this to True if you want to track the embedding progress.
        )
        return index
    # The entire indexing process is wrapped in a `try-except` block to catch 
    # and display any errors that may occur during the embedding or storing process.
    except Exception as e:
        print(f"Error in vector_database: {e}")
        return None


# Once the `VectorStoreIndex` is created, it holds the embedded document chunks (nodes) in vector form, which can now be searched and queried.
# 

# We can test out the functions by indexing the nodes we got from the previous step.
# 

# In[13]:


vectordb_index = vector_database(nodes)

if vectordb_index:
    print("Vector database created successfully.")
else:
    print("Failed to create vector database.")


# We can further inspect if the embeddings are all created properly by running the code block below:
# 

# In[14]:


vector_store = vectordb_index._storage_context.vector_store
node_ids = list(vectordb_index.index_struct.nodes_dict.keys())
missing_embeddings = False

for node_id in node_ids:
    embedding = vector_store.get(node_id)
    if embedding is None:
        print(f"Node ID {node_id} has a None embedding.")
        missing_embeddings = True
    else:
        print(f"Node ID {node_id} has a valid embedding.")

if missing_embeddings:
    print("Some node embeddings are missing. Please check the embedding generation step.")
else:
    print("All node embeddings are valid.")


# ### <a id='toc5_0_4_'></a>[Querying the indexed data](#toc0_)
# 

# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/sVghyREwXksA8DZwNleYPQ/RAG-Query.png" width="70%" alt="rag-query">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 
# After `Indexing` and `Storing` the data in vector form, the next step in a RAG workflow is `Querying`. This stage involves querying the indexed data to retrieve relevant information and generate responses based on the context of the query. The responses are powered by a language model that uses vector similarity to find and summarize the most relevant information from the indexed nodes.
# 

# #### Why do we need prompts and prompt engineering?
# 

# `Prompts` play a critical role in instructing the language model how to respond to a query or generate content based on the context. In this project, since we are asking the LLM to retrieve specific facts or answer questions based on LinkedIn profile data, we need to give clear and structured instructions through the prompt.
# 

# `Prompt Engineering` is the process of designing and optimizing these prompts to guide the model’s behavior. A well-designed prompt can significantly improve the quality and relevance of the responses, while a poorly designed prompt might lead to ambiguous or irrelevant answers.
# 

# In this project, we use prompts to:
# 
# **1.** Instruct the model on what type of response we want, whether it’s generating facts or answering a specific question. <br>
# **2.** Provide the model with the correct context from the indexed data, so the model can generate accurate responses based on the information it has. <br>
# **3.** Control how the model behaves when it doesn’t know the answer. For example, when the information is missing from the LinkedIn profile, we tell the model to respond with: "I don't know. The information is not available on the LinkedIn page."
# 

# #### What is PromptTemplate?
# 

# In LlamaIndex, a `PromptTemplate` is a flexible way to define the structure of a prompt. The `PromptTemplate` class allows you to create a customizable prompt by defining placeholders that are dynamically filled with relevant data at runtime.
# 
# - `{context_str}`: This placeholder is replaced with the context (i.e., the relevant document chunks or nodes that the model retrieves).
# - `{query_str}`: This placeholder is replaced with the user's query (i.e., the specific question the user asks).
# 

# #### Defining custom prompts
# 

# We use two different prompts for two different tasks:
# 
# **1.** Generating initial facts: When you want the assistant to provide general facts about the person’s career or education. <br>
# **2.** Answering user queries: When the user asks specific questions about the person, such as "What is their current job title?"
# 

# In[15]:


initial_facts_template = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""
initial_facts_prompt = PromptTemplate(template=initial_facts_template)


user_question_template = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context.If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
user_question_prompt = PromptTemplate(template=user_question_template)


# Then, we define a function to generate three interesting facts about a person's career or education, using the `initial_facts_prompt` defined earlier.
# 

# In this project, we will use `ibm/granite-3-8b-instruct` model.
# 
# The [Granite-3.0-8B Instruct model](https://www.ibm.com/granite/docs/?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Icebreaker+Bot+Lab-v1_1738621947) is an 8-billion parameter language model designed for tasks like summarization, text classification, multilingual dialog, and more. It builds upon the base model with fine-tuning on open-source and synthetic instruction datasets, offering structured chat capabilities. The model features advanced architecture elements like RoPE and SwiGLU, and supports 12 languages. Developed using IBM's Blue Vela supercomputing cluster, it is optimized for multilingual and instructional applications, allowing fine-tuning for specific languages and use cases.
# 

# In[16]:


def generate_initial_facts(index):
    """Generates 3 interesting facts about the person's career or education."""

    # Set the temperature for the model's response generation (controls creativity of the response).
    temperature = 0.0
    # Set the maximum number of new tokens (words) to generate in the response.
    max_new_tokens = 500
    additional_params = {
        "decoding_method": "sample",  # Sample from the probability distribution of tokens (instead of greedy decoding).
        "min_new_tokens": 1,          # Minimum number of tokens to generate.
        "top_k": 50,                  # Consider the top 50 most likely tokens at each step in the generation process.
        "top_p": 1,                   # Use nucleus sampling with a probability cutoff at 1 (i.e., consider all tokens).
    }

    # Initialize the WatsonxLLM instance for the ibm/granite-3-8b-instruct model
    watsonx_llm = WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url="https://us-south.ml.cloud.ibm.com", 
        project_id="skills-network",              
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        additional_params=additional_params,
    )
    
    # Create a query engine using the initial facts prompt
    query_engine = index.as_query_engine(
        streaming=False,                        # Disable streaming, wait for full response at once.
        similarity_top_k=5,                     # Use top 5 similar items from the index for query.
        llm=watsonx_llm,                        # Pass the Watsonx LLM with the IBM Granite model.
        text_qa_template=initial_facts_prompt    # Use a predefined prompt template to structure the LLM's output.
    )
    
    # Define a query that asks for 3 interesting facts about a person's career or education.
    query = "Provide three interesting facts about this person's career or education."
    
    # Execute the query using the query engine.
    response = query_engine.query(query)
    
    # Extract the actual generated facts from the response object.
    facts = response.response

    # Return the generated facts.
    return facts


# Next, we define a function to answer user-specific questions, such as "What is this person's current job title?" or "Where did this person work?"
# 

# In[17]:


from llama_index.llms.ibm import WatsonxLLM

def answer_user_query(index, user_query):
    """Answers the user's question using the vector database and the LLM."""

    try:
        # Set the temperature for controlling the randomness of the LLM's response.
        temperature = 0.0
        # Limit the number of new tokens generated in the response to 250.
        max_new_tokens = 250
        additional_params = {
            "decoding_method": "greedy",  # Greedy decoding for deterministic and predictable response.
            "min_new_tokens": 1,
            "top_k": 50,
            "top_p": 1,
        }
        
        # Initialize the WatsonxLLM instance for the ibm/granite-3-8b-instruct model
        watsonx_llm = WatsonxLLM(
            model_id="ibm/granite-3-8b-instruct",
            url="https://us-south.ml.cloud.ibm.com", 
            project_id="skills-network",              
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            additional_params=additional_params,
        )

        # Retrieve relevant nodes (chunks of data) from the index based on the user query.
        base_retriever = index.as_retriever(similarity_top_k=5)  # Fetch top 5 most relevant items from the index.
        source_nodes = base_retriever.retrieve(user_query)       # Retrieve relevant data chunks based on the query.

        # Build a context string by joining the text from each retrieved node.
        context_str = "\n\n".join([node.node.get_text() for node in source_nodes])
        
        # Create a query engine, specifying how the LLM should answer questions based on user input and the context.
        query_engine = index.as_query_engine(
            streaming=False,                        # Disable streaming, get the complete response all at once.
            similarity_top_k=5,                     # Use the top 5 similar items from the index for the query.
            llm=watsonx_llm,                        # Use the Watsonx LLM with the IBM Granite model.
            text_qa_template=user_question_prompt    # Provide a template to guide the LLM in forming the response.
        )
        
        # Execute the query with the user's question and return the LLM's answer.
        answer = query_engine.query(user_query)
        return answer
    
    except Exception as e:
        # Handle exceptions gracefully and log the error.
        print(f"Error in answer_user_query: {e}")
        return "Failed to get an answer."


# You can notice that in the second function (`answer_user_query`), we use a retriever to handle user-specific questions. This is a crucial part of the process when dealing with dynamic, variable queries (as opposed to predefined tasks such as "generating 3 facts about a person"). Let’s explore what a retriever is and why it’s essential in this context.
# 

# #### What is a retriever?
# 

# A `retriever` is responsible for searching through the indexed data and pulling out the most relevant chunks or nodes based on the user's query. It compares the vector embeddings of the user's query against the vector representations of the document chunks (nodes) that we previously embedded and stored in the vector database.
# 
# In other words, the retriever filters the data to find the most semantically relevant pieces of information to answer the user’s question. In this case, the retriever will return the top five nodes that are most similar to the query based on the vector similarity. You can adjust this number based on how detailed you want the responses to be.
# 
# 
# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/4hambo2ZiF_HcZ5M9Uxetg/Index%20Retrieval.png" width="70%" alt="basic-retrieval-system">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/examples/low_level/retrieval//">source</a></figcaption>
# </figure>
# 

# Let's test out the functions without vector database. The model might take about 1-2 minutes to generate responses.
# 

# In[18]:


initial_facts = generate_initial_facts(vectordb_index)
print("\nHere are 3 interesting facts about this person:")
print(initial_facts)


# In[19]:


user_query = "What is this person's current job title?"
response = answer_user_query(vectordb_index, user_query)
print(response)


# ### <a id='toc5_0_5_'></a>[Putting everything together: Building a chatbot interface](#toc0_)
# 

# Now that we have completed the key steps—**extracting LinkedIn data, splitting and indexing it, and retrieving answers based on user queries**—it’s time to bring everything together by building a chatbot interface. This interface will allow users to interact with the system by asking questions and getting responses based on the LinkedIn profile data.
# 
# We can use Python’s built-in `input()` function to create a simple text-based chatbot interface. This will allow the user to enter a LinkedIn profile URL and ask questions about the profile.
# 

# In[20]:


def chatbot_interface(index):
    """Provides a simple chatbot interface for user interaction."""
    print("\nYou can now ask more in-depth questions about this person. Type 'exit', 'quit' or 'bye' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        print("Bot is typing...", end='')
        sys.stdout.flush()
        time.sleep(1)  # Simulate typing delay
        print('\r', end='')
        
        response = answer_user_query(index, user_query)
        print(f"Bot: {response.response.strip()}\n")


# The following `process_linkedin` function is used to put everything together. 
# 

# In[21]:


def process_linkedin(linkedin_url, PROXYCURL_API_Key=None, mock=False):
    """
    Processes a LinkedIn URL, extracts data from the profile, and interacts with the user.

    Parameters:
    - linkedin_url (str): The LinkedIn profile URL to extract or load mock data from.
    - PROXYCURL_API_Key (str, optional): Proxycurl API key. Required if mock is False.
    - mock (bool, optional): If True, loads mock data from a premade JSON file instead of using the API.
    """
    try:
        # Extract the profile (with or without the API depending on the mock flag)
        profile_data = extract_linkedin_profile(linkedin_url, PROXYCURL_API_Key, mock=mock)
        
        if not profile_data:
            print("Failed to retrieve profile data.")
            return

        # Split the data into nodes
        nodes = split_profile_data(profile_data)
        
        # Store in vector database
        vectordb_index = vector_database(nodes)
        
        # Generate and display the initial facts
        initial_facts = generate_initial_facts(vectordb_index)
        
        print("\nHere are 3 interesting facts about this person:")
        print(initial_facts)
        
        # Start the chatbot interface
        chatbot_interface(vectordb_index)
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")


# We can test how the chatbot interface works with the vector database we constructed before. **After you complete, type 'exit', 'quit' or 'bye' to quit**. Some queries you can use to test the chatbot are:
# - When did this person start working at IBM?
# - Where did this person get their Master's degree?
# - What are some of this person's skills?
# 

# In[22]:


chatbot_interface(vectordb_index)


# You can also test the `process_linkedin` function with the other URL as shown below. **Note:** You need a valid ProxyCurl API key to run the code.
# 

# In[23]:


# Optional
profile_url = "https://www.linkedin.com/in/antoniocangiano/"
process_linkedin(profile_url, PROXYCURL_API_KEY, mock=False)


# ## <a id='toc6_0_'></a>[Exercises](#toc0_)
# 

# ### <a id='toc6_0_1_'></a>[Exercise 1 - Try with another LLM](#toc0_)
# 

# Watsonx.ai provides a group of available models. In the lab, we used IBM Granite models. Can you try to using another LLM such as `'meta-llama/llama-3-3-70b-instruct'`. [Here](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#) you can find more models you could choose from.
# 

# In[24]:


# Add your code here
model_id = 'meta-llama/llama-3-3-70b-instruct'


# <details>
#     <summary>Click here for a hint</summary>
#     
# ```python
# model_id = 'meta-llama/llama-3-3-70b-instruct'
# ```
# </details>
# 

# ### <a id='toc6_0_2_'></a>[Exercise 2 - Try with another LinkedIn profile (Optional)](#toc0_)
# 

# With the `process_linkedin` function, you can easily run the whole process with another LinkedIn Profile URL. You can try this profile : "https://www.linkedin.com/in/yoshuabengio/"
# 
# >**NOTE**: For this exercise, you will have to use ProxyCurl API.
# 

# In[25]:


# Add your code here
profile_url = "https://www.linkedin.com/in/yoshuabengio/"
process_linkedin(profile_url, PROXYCURL_API_KEY, mock=False)


# <details>
#     <summary>Click here for solution</summary>
# 
# ```python
# profile_url = "https://www.linkedin.com/in/yoshuabengio/"
# process_linkedin(profile_url, PROXYCURL_API_KEY, mock=False)
# ```
# 
# </details>
# 

# ## <a id='toc7_0_'></a>[Authors](#toc0_)
# 

# [Hailey Quach](https://www.haileyq.com/)
# 
# Hailey is a Data Scientist Intern at IBM. She is also an undergraduate student at Concordia University, Montreal
# 

# ## <a id='toc8_0_'></a>[Contributors](#toc0_)
# 

# [Ricky Shi](https://author.skills.network/instructors/ricky_shi)
# 
# [Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)
# 
# [Boyun Leung](https://author.skills.network/instructors/boyun_leung)
# 

# ## Change Log
# 

# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2025-02-11|1.3|Ricky Shi|Updated llm for exercise. Previous llm is deprecated.|
# |2024-10-22|1.2|Hailey Quach|Updated lab|
# |2024-09-23|1.1|Hailey Quach|Updated lab|
# |2024-09-19|1.0|Hailey Quach|Created lab|
# 

# Copyright © 2024 IBM Corporation. All rights reserved.
# 

# In[ ]:




