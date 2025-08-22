#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Creating a RAG application using Granite 3 LLM and LlamaIndex**
# 

# Estimated time needed: **30** minutes
# 

# ## Overview
# 

# Imagine you're developing an AI assistant tasked with providing expert-level, real-time answers to intricate user queries. The challenge lies in the fact that traditional models, while robust, often fail to incorporate the latest data or specific insights required to answer nuanced questions effectively.
# 
# With your Retrieval-Augmented Generation (RAG) application, created based on LlamaIndex, you'll overcome this limitation. Your assistant will  pull in the most relevant information from a broad array of data sources—whether it's the latest research, detailed reports, or up-to-date documentation—ensuring that every response is as precise and informed as possible. This capability is particularly crucial in fast-paced industries or research environments where having access to the most current and specific information can make all the difference.
# 
# This lab will guide you through the process of using LlamaIndex with its key components, culminating in a powerful RAG tool that can meet the demands of real-world applications where accuracy and context are paramount.
# 

# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rBprvgFTPOj5Fv4F5yMXkA/llamaindex.png" width="100%" alt="langchain">
# 

# In this lab, you will learn the steps of constructing a RAG application using LlamaIndex. RAG applications are at the cutting edge of AI, merging the capabilities of retrieval techniques with language models to create responses that are not only contextually aware but also incredibly relevant to the most current information available. 
# 
# By the end of this lab, you will be equipped to build a system that leverages the power of LlamaIndex to retrieve and integrate data into responses, ensuring that the content generated is always accurate and up-to-date.
# 

# ## __Table of Contents__
# 
# <ol>
#     <li><a href="#Overview">Overview</a></li>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
#             <li><a href="#Importing-required-libraries">Importing required libraries</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#RAG">RAG</a>
#         <ol>
#             <li><a href="#RAG-Stages">RAG Stages</a></li>
#             <li><a href="#Loading">Loading</a></li>
#             <li><a href="#Splitting">Splitting</a></li>
#             <li><a href="#Indexing">Indexing</a></li>
#             <li><a href="#Querying">Querying</a></li> 
#     </li>
# </ol>
# 
# <a href="#Exercises">Exercises</a>
# <ol>
#     <li><a href="#Exercise-1---Try-with-another-LLM">Exercise 1. Try with another LLM</a></li>
#     <li><a href="#Exercise-2---Work-on-another-document">Exercise 2. Work on another document</a></li>
# </ol>
# 

# ## Objectives
# 
# After completing this lab, you will be able to:
# 
# - Use LlamaIndex to construct a RAG application that retrieve information from documents.
# - Load, index, and retrieve data efficiently to ensure your RAG application accesses the most relevant information.
# - Enhance querying techniques with LlamaIndex for precise and context-aware responses.
# 

# ----
# 

# ## Setup
# 

# For this lab, we will be using the following libraries:
# 
# *   [`llama-index-llms-ibm`](https://docs.llamaindex.ai/en/stable/examples/llm/ibm_watsonx/) for communicating with watsonx.ai models using the LlamaIndex and watsonx.ai's LLMs API.
# *   [`llama-index-embeddings-ibm`](https://docs.llamaindex.ai/en/stable/examples/embeddings/ibm_watsonx/) for using watsonx.ai's embedding models.
# *   [`llama-index`](https://www.llamaindex.ai/) for using LlamaIndex framework relevant features.
# 

# ### Installing required libraries
# 
# The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:
# 
# **Note:** We are pinning the version here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.
# 
# This might take approximately 1-2 minutes. 
# 
# As we use `%%capture` to capture the installation, you won't see the output process. But after the installation completes, you will see a number beside the cell.
# 

# In[1]:


get_ipython().run_cell_magic('capture', '', '\n!pip install llama-index-llms-ibm==0.1.0 --user\n!pip install llama-index-embeddings-ibm==0.1.0 --user\n!pip install llama-index==0.10.65 --user\n')


# After you installat the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/build-a-hotdog-not-hotdog-classifier-guided-project/images/Restarting_the_Kernel.png" width="50%" alt="Restart kernel">
# 

# ### Importing required libraries
# 
# _We recommend you import all required libraries in one place (here):_
# 

# In[1]:


# Use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from llama_index.llms.ibm import WatsonxLLM
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.core import VectorStoreIndex


# ## RAG
# 

# Large Language Models (LLMs) are trained on vast datasets, but they don’t inherently include your specific data. Retrieval-Augmented Generation (RAG) addresses this limitation by integrating your data with the existing knowledge that LLMs possess. Throughout this documentation, you'll often encounter references to RAG. It's a key method used by query engines, chat engines, and agents to enhance their functionality.
# 
# In a RAG setup, your data is loaded, processed, and indexed for efficient querying. When a user submits a query, the system interacts with the index to narrow down your data to the most pertinent context. This relevant context, combined with the user's query, is then passed to the LLM, which generates a response based on this tailored information.
# 
# Whether you’re developing a chatbot or an intelligent agent, understanding RAG techniques is essential for effectively incorporating your data into the application.
# 

# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/NgRpgu9ZXi7goJw8W2eFFQ/basic-rag.png" width="100%" alt="langchain">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 

# ### RAG Stages
# 

# Within the RAG framework, there are five key stages, though in this lab, we'll focus on the first four. These stages are fundamental to most larger applications you might develop. The stages include:
# 
# - **Loading**: This involves bringing your data into your workflow, regardless of its source—be it text files, PDFs, websites, databases, or APIs. LlamaHub offers a wide array of connectors to facilitate this process.
# 
# - **Indexing**: This stage involves creating a data structure that enables efficient querying. For LLMs, this typically involves generating vector embeddings, which are numerical representations that capture the meaning of your data, along with various metadata strategies to ensure accurate and contextually relevant data retrieval.
# 
# - **Storing**: After indexing, it's usually important to save your index along with associated metadata to avoid the need for re-indexing in the future.
# 
# - **Querying**: Depending on your indexing strategy, there are multiple ways to utilize LLMs and LlamaIndex data structures for querying. This can include sub-queries, multi-step queries, and hybrid approaches.
# 
# - **Evaluation**: An essential stage in any workflow is evaluating how effective your approach is compared to others or when adjustments are made. Evaluation offers objective metrics to assess the accuracy, fidelity, and speed of your query responses.
# 

# <figure>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/uEQ_sNaMScgOgnNh6olKGg/stages.png" width="100%" alt="langchain">
#     <figcaption><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">source</a></figcaption>
# </figure>
# 

# #### Loading
# 

# In this project, we've provided a PDF file as an example data source. The RAG application we'll be building will retrieve information from this document.
# 

# To get started, let's download the PDF file into our current directory. This file will serve as the basis for our retrieval and generation tasks.
# 

# In[2]:


get_ipython().system('wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pfSOEORnYBZppsnhmZ1a8A/lora-paper.pdf"')


# Now that we have our PDF file in the directory, the next step is to load it into LlamaIndex. We'll use the `SimpleDirectoryReader` for this purpose.
# 
# `SimpleDirectoryReader` is a straightforward and efficient way to load data from a local file into LlamaIndex. It attempts to read all files found in the specified directory and processes them as text by default.
# 
# This method is particularly useful for quickly loading and preparing documents for further analysis and retrieval tasks in our RAG application.
# 

# In[3]:


documents = SimpleDirectoryReader(input_files=["lora-paper.pdf"]).load_data()


# Once the PDF is processed by the `SimpleDirectoryReader`, it will be converted into a `Document` object.
# 
# A Document in LlamaIndex acts as a container around any data source, whether it's a PDF, output from an API, or data retrieved from a database. By default, a `Document` stores text and includes additional attributes that provide more context and structure to the data.
# 
# Key attributes of a `Document` include:
# 
# - metadata: A dictionary of annotations that can be appended to the text, such as the document's title, author, or any other relevant information.
# - relationships: A dictionary containing references to other Documents or Nodes, enabling the creation of complex data structures and connections between various pieces of information.
# 

# Let's now take a look at the first page of the paper after it has been converted into a `Document` object. This will give us a sense of how the data has been structured and what information is available for retrieval.
# 

# In[4]:


documents[0]


# #### Splitting
# 

# Once you've loaded documents into your application, you might want to transform them to better suit your specific needs.
# 
# A common transformation is splitting a long document into smaller, more manageable chunks that can fit within your model's context window. LlamaIndex provides a built-in document splitter that makes it easy to split, combine, filter, and manipulate documents as needed.
# 
# The splitter works by dividing the document into nodes. A `Node` represents a "chunk" of a source document, whether that is a piece of text, an image, or another type of data. Like `Documents`, `Nodes` contain metadata and relationship information with other nodes, allowing for complex data structures and efficient information retrieval.
# 

# In the next step, we’ll configure the splitter using SentenceSplitter, setting the `chunk_size` to 500.
# 
# - chunk_size: This parameter specifies the maximum size of each node, ensuring they are as granular or broad as required for your application. By adjusting the `chunk_size`, you can control how much content each node holds, balancing the need for detailed information with the constraints of your model's context window.
# 

# Let’s proceed with setting up the splitter and see how it divides our document into manageable chunks.
# 

# In[5]:


splitter = SentenceSplitter(chunk_size=500)


# Let’s go ahead and apply the splitter to see how our document is transformed.
# 

# In[6]:


nodes = splitter.get_nodes_from_documents(documents)


# We can take a look at how many nodes we get.
# 

# In[7]:


len(nodes)


# We can also take a look at the first node's content.
# 

# In[8]:


node_metadata = nodes[0].get_content(metadata_mode=True)
print(str(node_metadata))


# #### Indexing
# 

# An `Index` is a crucial data structure in LlamaIndex that allows us to quickly retrieve relevant context in response to a user query. It's the core foundation for RAG use-cases, enabling efficient and accurate information retrieval.
# 
# At a high level, `Indexes` are built from `Documents`. These `Indexes` are then used to construct `Query Engines` and `Chat Engines`, which power question-and-answer interactions and conversational experiences over your data.
# 
# Under the hood, `Indexes` store data in `Node` objects, which represent the chunks of the original documents we created earlier. They also expose a `Retriever` interface, which can be configured and automated to enhance the retrieval process.
# 

# One critical step in indexing is converting your text data into vectors—a process known as embedding. These vectors allow the model to understand and compare the text data more effectively, facilitating the retrieval of relevant information.
# 
# To perform this embedding process, we need an embedding model. In the following code, we’ll demonstrate how to call an embedding model's API from watsonx.ai for downstream tasks, enabling the transformation of our document data into vectors that can be indexed and retrieved efficiently.
# 

# In[9]:


watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    truncate_input_tokens=3,
)


# Now we have the embedding model set up.
# 

# In LlamaIndex, one of the most commonly used methods for creating an index is the `VectorStoreIndex`. This method allows you to efficiently store and retrieve embedded document data.
# 
# When using `VectorStoreIndex`, you need to specify two key elements:
# 
# - Documents/Nodes: These are the pieces of text or data (in our case, the nodes we created earlier) that need to be embedded.
# - Embedding Model: This is the model that will transform the text data into vectors, enabling efficient retrieval.
# 
# Once you’ve specified the documents and the embedding model, `VectorStoreIndex` will automatically handle the embedding process. It converts the nodes into vector representations and stores these embeddings in a vector store. This setup ensures that your data is indexed in a way that supports fast and accurate retrieval for RAG applications.
# 
# Let’s proceed by specifying our documents and embedding model, and then creating the `VectorStoreIndex` to store our embeddings.
# 

# In[10]:


index = VectorStoreIndex(
    nodes=nodes, 
    embed_model=watsonx_embedding, 
    show_progress=True
)


# Now that we've created our `VectorStoreIndex`, we can leverage it to retrieve content based on similarity to a given query.
# 
# In LlamaIndex, you can set the index as a `Retriever`, which allows you to search for and retrieve the most relevant nodes (or document chunks) based on the similarity of their embeddings to the query.
# 
# For this example, we’ll configure the retriever to return the top 3 results for a query about "GPT-2." This means that when a query is made, the retriever will search through the vector store and return the three most relevant chunks of text that match the query.
# 

# In[11]:


base_retriever = index.as_retriever(similarity_top_k=3) # 3 for top 3 results

source_nodes = base_retriever.retrieve("GPT-2") # querying about GPT-2


# In[12]:


for node in source_nodes:
    # print(node.metadata)
    print(f"---------------------------------------------")
    print(f"Score: {node.score:.3f}")
    print(node.get_content())
    print(f"---------------------------------------------\n\n")


# #### Querying
# 

# At the querying step, we integrate a LLM to generate responses based on the retrieved content. The LLM takes the information retrieved by the VectorStoreIndex and generates a coherent and contextually relevant response to the user’s query.
# 

# IBM watsonx provides various foundation models that can be used for different tasks. 
# 
# This section shows how to use the [Granite model](https://newsroom.ibm.com/2023-09-28-IBM-Announces-Availability-of-watsonx-Granite-Model-Series,-Client-Protections-for-IBM-watsonx-Models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1_1720558829).
# 
# ### <a id='toc1_10_1_'></a>[Define the model](#toc0_)
# 
# In this project, you use `ibm/granite-3-8b-instruct`. This model will generate a response for the sample query: "What is Generative AI?"
# 
# The [Granite Gen-3 8 Billion instruct model (granite-3-8b-instruct)](https://www.ibm.com/new/ibm-granite-3-0-open-state-of-the-art-enterprise-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Granite+with+LangChain%3A+An+LLM+and+RAG+to+Answer+Questions_v1) is a new, instruction-tuned, dense decoder-only LLM. Trained using a novel two-phase method on over 12 trillion tokens of carefully vetted data across 12 different natural languages and 116 different programming languages, the developer-friendly Granite 3.0 8B Instruct is a workhorse enterprise model intended to serve as a primary building block for sophisticated workflows and tool-based use cases. Granite 3.0 8B Instruct matches leading similarly-sized open models on academic benchmarks while outperforming those peers on benchmarks for enterprise tasks and safety.
# 
# Previous generations of Granite models prioritized specialized use cases, excelling at domain-specific tasks across a diverse array of industries including finance, legal, code and academia. In addition to offering even greater efficacy in those arenas, IBM Granite 3.0 models match—and, in some cases, exceed—the general performance of leading open weight LLMs across both academic and enterprise benchmarks.
# 

# In[13]:


temperature = 0.1
max_new_tokens = 75
additional_params = {
    "decoding_method": "sample",
    "min_new_tokens": 1,
    "top_k": 50,
    "top_p": 1,
}

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    temperature=temperature,
    max_new_tokens=max_new_tokens,
    additional_params=additional_params,
)


# In[14]:


response = watsonx_llm.complete("What is a Generative AI?")
print(response)


# Now that we've seen the LLM in action and confirmed that it works well, the next step is to integrate this LLM into our query engine.
# 
# By doing so, the LLM will not only generate responses but will also work seamlessly with the retrieval system we've set up. This means that when a user submits a query, the query engine will first retrieve the most relevant information from the indexed documents and then use the LLM to generate a comprehensive and context-aware response.
# 

# In[15]:


query_engine = index.as_query_engine(
  streaming=False, 
  similarity_top_k=7, 
  llm=watsonx_llm
)


# With the LLM now integrated into our query engine, we can utilize the full power of our RAG application. This means we can issue queries, and the engine will both retrieve the most relevant information and generate well-informed responses using the LLM.
# 
# Below are two examples demonstrating how to query the engine:
# 

# In[16]:


response = query_engine.query("What is the lora paper about?")
print(str(response))


# In[17]:


response = query_engine.query("List all the evaluation datasets that where used in the lora paper. Only consider the paper.")
print(str(response))


# # Exercises
# 

# ### Exercise 1 - Try with another LLM
# 
# Watsonx.ai provides a group of available models. In the lab, we are using `ibm/granite-3-8b-instruct`. Can you try to using another LLM such as `'meta-llama/llama-3-3-70b-instruct'`. [Here](https://ibm.github.io/watsonx-ai-python-sdk/fm_model.html#) you can find more models you could choose from.
# 

# In[18]:


# Add your code here
#model_id = 'meta-llama/llama-3-3-70b-instruct'

watsonx_embedding = WatsonxEmbeddings(
    model_id="meta-llama/llama-3-3-70b-instruct'",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    truncate_input_tokens=3,
)


# In[20]:


'''index = VectorStoreIndex(
    nodes=nodes, 
    embed_model=watsonx_embedding, 
    show_progress=True
)'''


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# model_id = 'meta-llama/llama-3-3-70b-instruct'
# ```
# 
# </details>
# 

# ### Exercise 2 - Work on another document
# 

# You are welcome to use another document to practice. Another document has also been prepared here. Can you load this document and make the LLM read it for you? <br>
# Here is the URL to the document: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t27UNbGN9hNoWnS13mJP1A/companypolicies.txt
# 

# In[21]:


# Add your code here
get_ipython().system('wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t27UNbGN9hNoWnS13mJP1A/companypolicies.txt"')

documents = SimpleDirectoryReader(input_files=["companypolicies.txt"]).load_data()


# <details>
#     <summary>Click here for a hint</summary>
#     
# ```python
# !wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t27UNbGN9hNoWnS13mJP1A/companypolicies.txt"
# 
# documents = SimpleDirectoryReader(input_files=["companypolicies.txt"]).load_data()
# ```
# </details>
# 

# ## Authors
# 

# [Kang Wang](https://author.skills.network/instructors/kang_wang) is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.
# 

# ### Other Contributors
# 

# [Fateme Akbari](https://author.skills.network/instructors/fateme_akbari) is a Ph.D. candidate in Information Systems at McMaster University with demonstrated research experience in Machine Learning and NLP.
# 
# [Wojciech Fulmyk](https://author.skills.network/instructors/wojciech_fulmyk) is a data scientist at the Ecosystems Skills Network at IBM. He is also a Ph.D. candidate in Economics at the University of Calgary.
# 
# [Faranak Heidari](https://www.linkedin.com/in/faranakhdr/) is a data scientist at the Ecosystems Skills Network at IBM. She is also a Ph.D. candidate at the University of Toronto.
# 
# 

# ## Change Log
# 

# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2025-02-04|0.4|Faranak Heidari|Review and Update exercise to new Llama|
# |2024-10-17|0.4|Faranak Heidari|Review and Update to Gen3|
# |2024-08-23|0.3|Wojciech Fulmyk|Review|
# |2024-08-19|0.2|Fateme Akbari|Review|
# |2024-08-15|0.1|Kang Wang|Create the lab|
# 

# Copyright © 2024 IBM Corporation. All rights reserved.
# 

# In[ ]:




