#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # <a id='toc1_'></a>[Building a RAG system for web data using Llama 3.1-405b](#toc0_)
# 

# Note: This project builds upon the [Create a LangChain RAG system for web data in Python using Llama 3.1-405b in watsonx.ai](https://developer.ibm.com/tutorials/awb-create-langchain-rag-system-web-data-llama405b-watsonx/) tutorial authored by Erika Russi.
# 
# Estimated time needed: **30** minutes
# 
# In this guided project, we will use LangChain and `meta-llama/llama-3-405b-instruct` to walk through a step-by-step Retrieval Augmented Generation ([RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Building+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1722347683)) example in Python.
# 

# **Table of contents**<a id='toc0_'></a>    
# 
#   - [Introduction](#toc1_1_)    
#   - [What does this guided project do?](#toc1_2_)    
#   - [Objectives](#toc1_3_)    
#   - [Background](#toc1_4_)    
#     - [What is Large Language Model (LLM)?](#toc1_4_1_)    
#     - [What is IBM watsonx?](#toc1_4_2_)    
#     - [Why watsonx vs other cloud platforms?](#toc1_4_3_)    
#     - [What is LangChain?](#toc1_4_4_)    
#     - [What is Llama 3.1-405b?](#toc1_4_5_)    
#     - [What is Retrieval Augmented Generation (RAG)?](#toc1_4_6_)    
#     - [More about RAG and LangChain](#toc1_4_7_)    
#   - [Setup](#toc1_5_)    
#     - [Installing required libraries](#toc1_5_1_)    
#   - [Watsonx API credentials and project_id](#toc1_6_)    
#   - [Index the URLs to create the knowledge base](#toc1_7_)    
#   - [Set up a retriever](#toc1_8_)    
#   - [Generate a response with a generative model](#toc1_9_)    
#   - [Exercises](#toc1_10_)    
#     - [Exercise 1 - ask more questions](#toc1_10_1_)     
# 

# ## <a id='toc1_1_'></a>[Introduction](#toc0_)
# 
# Imagine you have several web pages, and want to extract information from them. You could read each page and take notes, but that would be time-consuming. Instead, you can use a RAG system to help you. RAG systems combine the power of a large language model (LLM) with a retrieval system to provide context to the LLM. This allows you to ask questions about the content of the web pages and get answers quickly.
# 
# 
# ## <a id='toc1_2_'></a>[What does this guided project do?](#toc0_)
# 
# We will set up a local RAG system for several IBM products. We will fetch content from web pages, making up a knowledge base from which we will provide context to Meta's Llama 3.1-405b LLM to answer some questions about these IBM products.
# 
# 
# ## <a id='toc1_3_'></a>[Objectives](#toc0_)
# 
# After completing this lab, you will be able to:
# 
# - Understand how to set up and configure LangChain for advanced language modeling tasks.
# - Learn to use Llama 3.1-405b on watsonx.ai to enhance your language model's capabilities.
# - Develop a RAG system to generate context-aware, real-time responses from web data.
# 

# ## <a id='toc1_4_'></a>[Background](#toc0_)
# 
# ### <a id='toc1_4_1_'></a>[What is Large Language Model (LLM)?](#toc0_)
# 
# [Large language models](https://www.ibm.com/topics/large-language-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) are a category of foundation models trained on immense amounts of data making them capable of understanding and generating natural language and other types of content to perform a wide range of tasks.
# 
# ### <a id='toc1_4_2_'></a>[What is IBM watsonx?](#toc0_)
# 
# [IBM watsonx](https://www.ibm.com/watsonx?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) is a suite of artificial intelligence (AI) tools and services that are designed to help developers build and deploy AI-driven applications. watsonx provides a range of APIs and tools that make it easy to integrate AI capabilities into applications, including natural language processing, computer vision, and speech recognition.
# 
# **Enterprises  turn to watsonx because it is:**
# 
# - **Open**: Based on open technologies that provide a variety of models to cover enterprise use cases and support compliance initiatives.
# - **Targeted**: Targeted to specific enterprise domains like HR, customer service or IT operations to unlock new value.
# - **Trusted**: Designed with principles of transparency, responsibility, and governance, so that you can manage  ethical and accuracy concerns.
# - **Empowering**: Go beyond being an AI user and become an AI value creator, owning the value your models create.
# 
# ### <a id='toc1_4_3_'></a>[Why watsonx vs other cloud platforms?](#toc0_)
# 
# - **Infrastructure**: watsonx offers hybrid, multi-cloud option for model deployment.
# - **Models**: watsonx's ability to deliver on-premise. watsonx offers deployment flexibility and additional safeguards when working with proprietary data not suited for a 3rd party cloud.
# - **Platform**: watsonx is not just watsonx.ai – it is also watsonx.data and watsonx.governance, adding data control/management and data/AI governance. This gives clients an infrastructure that allows them to develop/deploy and govern the data and the AI model being used.
# 
# 
# ### <a id='toc1_4_4_'></a>[What is LangChain?](#toc0_)
# 
# [LangChain](https://www.ibm.com/topics/langchain?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) is an open source orchestration framework for the development of applications using LLMs. Available in both Python- and JavaScript-based libraries, LangChain’s tools and APIs simplify the process of building LLM-driven applications like [chatbots](https://www.ibm.com/topics/chatbots?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) and [virtual agents](https://www.ibm.com/topics/virtual-agent?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829). 
# 
# ### <a id='toc1_4_5_'></a>[What is Llama 3.1-405b?](#toc0_)
# 
# [On Tuesday, July 23, 2024, Meta announced the launch of the Llama 3.1 collection of multilingual large language models (LLMs)](https://www.ibm.com/blog/meta-releases-llama-3-1-models-405b-parameter-variant/?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829). Llama 3.1 comprises both pretrained and instruction-tuned text in/text out open source generative AI models in sizes of 8B, 70B and—for the first time—405B parameters. Llama-3.1-405B is one of the world’s largest and most powerful open models. 
# 
# Powered by 405 billion parameters, this model specializes in generating synthetic data, distilling knowledge into smaller models, domain-specific fine-tuning and evaluating other models’ responses. Its performance either matches or surpasses leading large language models in a variety of tests assessing undergraduate level knowledge, graduate level reasoning, math problem solving, reading comprehension, and more.
# 
# Unlike its closed source peers, Llama-3.1-405B is open source, meaning it can be built upon and improved by the broader community.
# 
# More information about Llama 3.1-405b can be found [here](https://ai.meta.com/blog/meta-llama-3-1/).
# 
# ### <a id='toc1_4_6_'></a>[What is Retrieval Augmented Generation (RAG)?](#toc0_)
# 
# [RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) is a technique in natural language processing (NLP) that combines information retrieval and generative models to produce more accurate, relevant, and contextually aware responses.
# 
# ### <a id='toc1_4_7_'></a>[More about RAG and LangChain](#toc0_)
# 
# In traditional language generation tasks, LLMs, such as OpenAI’s GPT (Generative Pre-trained Transformer) or IBM’s Granite Models, are used to construct responses based on an input prompt. However, these models can struggle to produce responses that are contextually relevant, factually accurate, or up-to-date. The models may not know the latest information about IBM products.
# 
# RAG applications address this limitation by incorporating a retrieval step before response generation. During retrieval, [vector search](https://www.ibm.com/topics/vector-search?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) can be used to identify contextually pertinent information, such as relevant information or documents from a large corpus of text, typically stored in a [vector database](https://www.ibm.com/topics/vector-database?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829). Finally, an LLM is used to generate a response based on the retrieved context. RAG is an affordable and simple alternative to [fine-tuning](https://www.ibm.com/topics/fine-tuning?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) a model for text-generation artificial intelligence tasks.
# 
# LangChain is a powerful, open-source framework that facilitates the development of applications using LLMs for various NLP tasks. In the context of RAG, LangChain plays a critical role by combining the strengths of retrieval-based methods and generative models to enhance the capabilities of NLP systems.
# 

# ## <a id='toc1_5_'></a>[Setup](#toc0_)
# 
# For this lab, we will be using the following libraries:
# 
# 
# *   [`langchain`](https://pypi.org/project/langchain/): Building applications with LLMs through composability.
# *   [`ibm-watsonx-ai`](https://pypi.org/project/ibm-watsonx-ai/): `ibm-watsonx-ai` is a library that allows to work with watsonx.ai service on IBM Cloud and IBM Cloud for Data. Train, test, and deploy your models as APIs for application development and share with colleagues using this python library.
# *   [`langchain-ibm`](https://pypi.org/project/langchain-ibm/): This package provides the integration between LangChain and IBM watsonx.ai through the ibm-watsonx-ai SDK.
# *   [`unstructured`](https://pypi.org/project/unstructured/): A library that prepares raw documents for downstream ML tasks.
# *   [`ibm-watson-machine-learning`](https://pypi.org/project/ibm-watson-machine-learning/): A library that allows to work with Watson Machine Learning service on IBM Cloud and IBM Cloud for Data. Train, test, and deploy your models as APIs for application development and share with colleagues using this python library.
# 

# ### <a id='toc1_5_1_'></a>[Installing required libraries](#toc0_)
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You must run the following cell__ to install them. Please wait until it completes.
# 
# This step could take **several minutes**, please be patient.
# 
# **NOTE**: If you encounter any issues, please restart the kernel and run again.  You can do that by clicking the **Restart the kernel** icon.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/crvBKBOkg9aBzXZiwGEXbw/Restarting-the-Kernel.png" width="50%" alt="Restart kernel">
# 

# In[1]:


get_ipython().run_line_magic('pip', 'install langchain==0.2.6 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain_chroma==0.1.2 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain-community==0.2.6 | tail -n 1')
get_ipython().run_line_magic('pip', 'install ibm-watsonx-ai==1.0.10 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain_ibm==0.1.11 | tail -n 1')
get_ipython().run_line_magic('pip', 'install unstructured==0.15.0 | tail -n 1')
get_ipython().run_line_magic('pip', 'install ibm-watson-machine-learning==1.0.360 | tail -n 1')


# In[2]:


#import required libraries
import os

from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


import warnings
warnings.filterwarnings('ignore')


# ## <a id='toc1_6_'></a>[Watsonx API credentials and project_id](#toc0_)
# 
# This section provides you with the necessary credentials to access the watsonx API.
# 
# **Please note:**
# 
# In this lab environment, you don't need to specify the api_key, and the project_id is pre_set as "skills-network", but if you want to use the model locally, you need to go to [watsonx](https://www.ibm.com/watsonx?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Build+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1727723829) to create your own keys and ID.
# 

# In[3]:


from ibm_watsonx_ai import Credentials
import os


credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )

project_id = "skills-network"


# ## <a id='toc1_7_'></a>[Index the URLs to create the knowledge base](#toc0_)
# 
# We’ll index our IBM products specific pages from URLs to create a knowledge base as a vectorstore. The content from these URLs will be our data sources and context for this exercise. The context will then be provided to an LLM to answer any questions we have about the IBM products.
# 
# The first step to building vector embeddings is to clean and process the raw dataset. This may involve the removal of noise and standardization of the text. For our example, we won’t do any cleaning since the text is already cleaned and standardized.
# 
# First, let's establish `URLS_DICTIONARY`. `URLS_DICTIONARY` is a dict that helps us map the URLs from which we will be extracting the content. Let's also set up a name for our collection: `ibm_products`.
# 
# Next, let's load our documents for the list of URLs we have. We'll print a sample document at the end to see how it's been loaded. 
# 

# In[4]:


import requests

class Document:
    def __init__(self, metadata, page_content):
        self.metadata = metadata
        self.page_content = page_content

URLS_DICTIONARY = {
    "watsonx_wiki": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/PWMJ9-Npq9FYNSWrrf99YQ/watsonx.txt",
    "ibm_cloud_wiki": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/wxekgOAVRH71dO92DEbwfQ/ibm-cloud.txt",
}
COLLECTION_NAME = "ibm_products"

documents = []

for name, url in URLS_DICTIONARY.items():
    print(f"Loading from {url}")
    response = requests.get(url)

    if response.status_code == 200:
        data = {
            'metadata': {"source": url, "name": name},
            'page_content': response.text
        }

        documents.append(Document(metadata=data['metadata'], page_content=data['page_content']))
        print(f"Loaded from {url}")
    else:
        print(f"Failed to retrieve content from {url}")


print(documents[0].metadata)
print(documents[0].page_content)


# Based on the sample document, it looks like there's a lot of white space and new line characters that we can get rid of. Let's clean that up and add some metadata to our documents, including an ID number and the source of the content.
# 

# In[5]:


doc_id = 0
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split()) # remove white space

    doc.metadata["id"] = doc_id #make a document id and add it to the document metadata

    print(doc.metadata)
    doc_id += 1

# Let's see how our sample document looks now after we cleaned it up.
display(documents[1].metadata)
display(documents[1].page_content)


# We need to split up our text into smaller, more manageable pieces known as "chunks". LangChain's `RecursiveCharacterTextSplitter` takes a large text and splits it based on a specified chunk size using a predefined set of characters. In order, the default characters are:
# 
# - "\n\n" - two new line characters
# - "\n" - one new line character
# - " " - a space
# - "" - an empty character
# 
# The process starts by attempting to split the text using the first character, "\n\n." If the resulting chunks are still too large, it moves to the next character, "\n," and tries splitting again. This continues with each character in the set until the chunks are smaller than the specified chunk size.
# 

# In[6]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# Next, we choose an embedding model to be trained on our IBM products dataset. The trained embedding model is used to generate embeddings for each data point in the dataset. For text data, popular open-source embedding models include Word2Vec, GloVe, FastText or pretrained transformer-based models like BERT or RoBERTa. OpenAIembeddings may also be used by leveraging the OpenAI embeddings API endpoint, the `langchain_openai` package and getting an `openai_api_key`, however, there is a cost associated with this usage.
# 
# Unfortunately, because the embedding models are so large, vector embedding often demands significant computational resources, like a GPU. We can greatly lower the costs linked to embedding vectors, while preserving performance and accuracy by using WatsonxEmbeddings. We'll use the IBM embeddings model, Slate, an encoder-only (RoBERTa-based) model, which while not generative, is fast and effective for many NLP tasks.
# 
# Alternatively, we can use the [Hugging Face embeddings models](https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/#embedding-models) via LangChain.
# 

# In[7]:


embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    project_id=project_id,
    )


# Let's load our content into a local instance of a vector database, using Chromadb.
# 

# In[8]:


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)


# Let's do a quick search of our vector database to test it out! Using `similarity_search_with_score` allows us to return the documents and the distance score of the query to them. The returned distance score is Euclidean distance. Therefore, a lower score is better.
# 
# You can adjust `k` to fetch the number of results you want to return.
# 

# In[9]:


query = "What is IBM?"
search = vectorstore.similarity_search_with_score(query, k=4)
search


# ## <a id='toc1_8_'></a>[Set up a retriever](#toc0_)
# 
# We'll set up our vector store as a retriever. The retrieved information from the vector store serves as additional context or knowledge that can be used by a generative model.
# 
# You can also specify search kwargs like `k` (the number of documents to return (Default: 4)) to use when doing retrieval.
# 

# In[10]:


retriever = vectorstore.as_retriever(search_kwargs={'k':2})


# ## <a id='toc1_9_'></a>[Generate a response with a generative model](#toc0_)
# 
# Finally, we’ll generate a response. The generative model (like GPT-4 or IBM Granite) uses the retrieved information to produce a more accurate and contextually relevant response to our questions about IBM products.
# 
# First, we'll establish the LLM we're going to use to generate the response. For this tutorial, we'll use Llama 3.
# 
# The available model parameters can be found [here](https://ibm.github.io/watson-machine-learning-sdk/model.html#enums).
# 
# For more information on model parameters and what they mean, see [Foundation model parameters: decoding and stopping criteria](https://www.ibm.com/docs/en/watsonx/saas?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Building+a+RAG+system+for+web+data+using+Llama+3.1-405b_v1_1722347683&topic=lab-model-parameters-prompting).
# 

# In[11]:


model_id = "meta-llama/llama-3-405b-instruct"

parameters = {
    GenParams.DECODING_METHOD: 'greedy',
    GenParams.MIN_NEW_TOKENS: 10,
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.REPETITION_PENALTY:1,
    GenParams.RETURN_OPTIONS: {'input_tokens': True,'generated_tokens': True, 'token_logprobs': True, 'token_ranks': True, }
}

# instantiate the LLM
llm = WatsonxLLM(
    model_id=model_id,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)


# We'll set up a `prompt template` to ask multiple questions. The "context" will be derived from our retriever (our vector database) with the relevant documents and the "question" will be derived from the user query.
# 

# In[12]:


template = """Generate a summary of the context that answers the question. Explain the answer in multiple steps if possible. 
Answer style should match the context. Ideal Answer Length 2-3 sentences.\n\n{context}\nQuestion: {question}\nAnswer:
"""
prompt = ChatPromptTemplate.from_template(template)


# Let's set up a helper function to format the docs accordingly:
# 

# In[13]:


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Next, we can set up a chain with our context, our prompt, and our LLM model. We'll use `StrOutputParser` for parsing the results. The generative model processes the augmented context along with the user's question to produce a response.
# 

# In[14]:


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# We can now ask questions:
# 

# In[15]:


import pprint

pprint.pprint(chain.invoke("What is watsonx?"), width=120) 


# In[16]:


pprint.pprint(chain.invoke("Tell me about IBM"), width=120)


# ## <a id='toc1_10_'></a>[Exercises](#toc0_)
# 
# ### <a id='toc1_10_1_'></a>[Exercise 1 - ask more questions](#toc0_)
# 
# Please check the answer to the following topics: 
# 
# `What is IBM cloud?`
# 

# In[17]:


# Type your code here
pprint.pprint(chain.invoke("What is IBM cloud?"), width=120)


# In[18]:


pprint.pprint(chain.invoke("What is the latest ai innotion"), width=120)


# <details>
#     <summary>Click here for Solution
#     </summary>
# 
# ```python
# pprint.pprint(chain.invoke("What is IBM cloud?"), width=120) 
# ```
# </details>
# 

# ## <a id='toc1_11_'></a>[Authors](#toc0_)
# 
# 
# [Ricky Shi](https://author.skills.network/instructors/ricky_shi)
# 

# ## <a id='toc1_12_'></a>[Contributors](#toc0_)
# 
# [Wojciech "Victor" Fulmyk](https://www.linkedin.com/in/wfulmyk)
# 
# [Hailey Quach](https://www.haileyq.com/)
# 

# Copyright © 2024 IBM Corporation. All rights reserved.
# 
