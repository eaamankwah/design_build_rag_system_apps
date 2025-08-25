#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **YouTube Summarization and Q&A with Granite, LangChain and RAG**
# 

# Estimated time needed: **60** minutes
# 

# ## Introduction
# 

# In this lab, you’ll dive into the exciting world of YouTube video analysis, tackling the problem of extracting and summarizing key information from lengthy video transcripts. With the explosion of video content online, it's impractical to manually sift through hours of footage to find important details. This lab will empower you to automate this process by transforming dense transcripts into clear, concise summaries. You’ll harness the power of FAISS for pinpointing relevant video segments and LangChain for crafting a robust question-answering system. By the end of this lab, you'll not only streamline how you interact with video content but also develop skills that are crucial for building intelligent systems capable of navigating vast amounts of multimedia data. Get ready to revolutionize the way you process and understand video content!
# 

# ![gp.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rcjilqoYQBytaou94o6PUg/gp.png)
# 

# ## __Table of Contents__
# 
# <ol>
#     <li><a href="#introduction">Introduction</a></li>
#     <li><a href="#objectives">Objectives</a></li>
#     <li>
#         <a href="#setup">Setup</a>
#         <ol>
#             <li><a href="#installing-required-libraries">Installing Required Libraries</a></li>
#             <li><a href="#importing-required-libraries">Importing Required Libraries</a></li>
#         </ol>
#     </li>
#     <li><a href="#extracting-youtube-transcripts">Extracting YouTube Transcripts</a></li>
#     <li><a href="#processing-the-transcript">Processing the Transcript</a></li>
#     <li><a href="#chunking-the-transcript">Chunking the Transcript</a></li>
#     <li>
#         <a href="#setting-up-watsonx-model">Setting up watsonx Model</a>
#         <ol>
#             <li><a href="#credentials-setup">Credentials Setup</a></li>
#             <li><a href="#defining-parameters-for-watsonx-model">Defining Parameters for watsonx Model</a></li>
#             <li><a href="#initializing-watsonx-llm">Initializing watsonx LLM</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#embedding-and-similarity-search">Embedding and Similarity Search</a>
#         <ol>
#             <li><a href="#embedding-the-transcript-chunks">Embedding the Transcript Chunks</a></li>
#             <li><a href="#implementing-faiss-for-similarity-search">Implementing FAISS for Similarity Search</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#summarizing-the-transcript-with-llmchain">Summarizing the Transcript with LLMChain</a>
#         <ol>
#             <li><a href="#define-the-prompt-template">Define the Prompt Template</a></li>
#             <li><a href="#instantiate-the-llmchain">Instantiate the LLMChain</a></li>
#             <li><a href="#generate-the-summary">Generate the Summary</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#retrieving-relevant-context-and-generating-answers">Retrieving Relevant Context and Generating Answers</a>
#         <ol>
#             <li><a href="#retrieving-relevant-context">Retrieving Relevant Context</a></li>
#             <li><a href="#creating-the-prompt-template">Creating the Prompt Template</a></li>
#             <li><a href="#setting-up-the-llmchain">Setting Up the LLMChain</a></li>
#             <li><a href="#generating-an-answer">Generating an Answer</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#exercise">Exercise</a>
#         <ol>
#             <li><a href="#exercise-1-ask-more-questions">Exercise: 1 Ask More Questions</a></li>
#             <li><a href="#exercise-2-try-a-different-video">Exercise: 2 Try a Different Video</a></li>
#         </ol>
#     </li>
#     <li><a href="#conclusion">Conclusion</a></li>
#     <li><a href="#next-steps">Next Steps</a></li>
#     <li><a href="#authors">Authors</a></li>
#     <li><a href="#contributors">Contributors</a></li>
# </ol>
# 

# ## Objectives
# 
# After completing this lab you will be able to:
# 
# - Use the `youtube-transcript-api` to retrieve transcripts from YouTube videos.
# - Utilize the `LangChain` framework to create concise summaries of the video content.
# - Implement similarity search using `FAISS` to find relevant content based on user queries.
# - Develop and utilize `QA chains` to generate answers to specific questions based on the video content.
# - Enhance the quality and relevance of responses using `retrieval-augmented generation (RAG)` techniques.
# 

# ----
# 

# ## Setup
# 

# For this lab, we will be using the following libraries:
# 
# *  `youtube-transcript-api` for extracting transcripts from YouTube videos.
# *  `faiss-cpu` for efficient similarity search.
# *  `langchain` and `langchain-community` for text processing and language models.
# *  `ibm-watsonx-ai` and `langchain_ibm` for integrating IBM Watson services.
# 

# ### Installing Required Libraries
# 
# We need to install a few libraries to run the code. Run the following commands to install the required libraries:
# 

# In[1]:


#!pip install youtube-transcript-api
get_ipython().system('pip install youtube-transcript-api==0.6.2')
get_ipython().system('pip install faiss-cpu==1.8.0')
get_ipython().system('pip install langchain==0.2.6 | tail -n 1')
get_ipython().system('pip install langchain-community==0.2.6 | tail -n 1')
get_ipython().system('pip install ibm-watsonx-ai==1.0.10 | tail -n 1')
get_ipython().system('pip install langchain_ibm==0.1.8 | tail -n 1')


# ### Importing Required Libraries
# 
# We will import all the necessary libraries for the project here. This includes libraries for transcript extraction, text processing, and embedding models.
# 
# 

# In[2]:


from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re

# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# ## What is RAG?
# 
# RAG (Retrieval-Augmented Generation) is an AI approach that combines retrieval of relevant information from external data sources with a generative model. It first retrieves documents or facts and then uses them to generate more informed, contextually accurate responses. This improves the quality and relevance of generated text, especially for tasks requiring real-time or domain-specific knowledge.
# 

# ![Rag Basics.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Cu3ViPmsJajdRYnYcTpYCg/Rag%20Basics.png)
# 

# ## What are RAG Stages?
# 
# ### RAG Stages Applied to AI YouTube Summarizer & Q&A System
# 
# - **Loading**: Extract video transcripts from YouTube using a pre-built API, preparing them as the initial data source for processing.
# 
# - **Splitting**: Break the long video transcript into smaller, more manageable sections or chunks to facilitate easier retrieval and summarization.
# 
# - **Indexing**: Organize the transcript chunks efficiently using FAISS (Facebook AI Similarity Search) to ensure quick and relevant retrieval for question-answering.
# 
# - **Storing**: Save the indexed video transcript data for quick access during user queries and future summarization tasks.
# 
# - **Querying**: Retrieve relevant transcript sections based on user questions to generate specific answers from the video content, eliminating the need to watch the entire video.
# 
# - **Evaluating**: Generate and refine summaries and Q&A responses by assessing the retrieved information for accuracy, relevance, and clarity, providing users with concise and valuable insights.
# 
# By following these stages, the system transforms raw YouTube video transcripts into interactive summaries and real-time Q&A capabilities, boosting engagement and saving time.
# 

# ![RAG_stages.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/jumDx7nEWk4DQ8gxg58rYw/RAG-stages.png)
# 

# ## Extracting YouTube Transcripts
# 
# We use the `YouTubeTranscriptApi` to retrieve transcripts for a YouTube video. The `get_transcript` function fetches both manually generated and automatically generated transcripts for the given video ID.
# 
# `Automatic vs. Manual Transcripts`: YouTube provides both auto-generated and manually uploaded transcripts. Manually created transcripts are generally more accurate, so the function prioritizes these when available.
# 
# How Transcripts Are Fetched: The `list_transcripts` function returns all available transcripts for a video. We loop through them to find the most accurate one. If a manually created transcript is found, it overrides the auto-generated one.
# 
# **Function**: `get_video_id`
# 
# The `get_video_id` function extracts the video ID from a YouTube URL.
# 
# - **Input:** A YouTube video URL.
# - **Output:** The video ID if the URL matches the expected pattern, otherwise `None`.
# 
# **YouTube URL Format**
# 
# A typical YouTube URL format is: https://www.youtube.com/watch?v=VIDEO_ID
# 
# Where `VIDEO_ID` consists of 11 alphanumeric characters, including hyphens or underscores.
# 
# **Function**: `get_transcript`
# The get_transcript function retrieves the transcript for a given YouTube video.
# 
# - **Input**: A YouTube video URL.
# - **Output**: The transcript as a list of dictionaries, where each dictionary represents a segment of the transcript.
# 

# In[3]:


def get_video_id(url):    
    # Define a regular expression pattern to match YouTube video URLs
    # The pattern captures 11 alphanumeric characters (plus hyphen or underscore) after '?v='
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    
    # Search the provided URL for the pattern
    match = re.search(pattern, url)
    
    # If a match is found, return the captured video ID group
    # Otherwise, return None
    return match.group(1) if match else None


# In[4]:


def get_transcript(url):
    video_id = get_video_id(url)
    # Fetches the list of available transcripts for the given YouTube video
    srt = YouTubeTranscriptApi.list_transcripts(video_id)

    transcript = ""
    for i in srt:
        # Check if the transcript is auto-generated
        if i.is_generated:
            # If no transcript has been set yet, use the auto-generated one
            if len(transcript) == 0:
                transcript = i.fetch()
        else:
            # If a manually created transcript is found, use it (overrides auto-generated)
            transcript = i.fetch()

    return transcript


# **Transcript Structure**
# 
# The transcript is represented as a list of dictionaries. Each dictionary contains:
# 
# - **text**: The spoken content.
# - **start**: The starting time of the segment in seconds.
# - **duration**: The duration of the segment in seconds.
# 

# In[5]:


# Retrieve the transcript for the specified YouTube video URL
transcript = get_transcript("https://www.youtube.com/watch?v=BXPqj6nKQ5c")
#transcript = get_transcript("https://www.youtube.com/watch?v=kEOCrtkLvEo&t=24s")
#transcript = ytt_api.fetch("osKyvYJ3PRM")
#transcript = get_transcript("XzzdFaIOdyY")

# Display the first 10 entries of the transcript
# Each entry is a dictionary containing 'text', 'start', and 'duration'
transcript[:10]


# ## Processing the Transcript
# 
# We now process the fetched transcript into a readable format by extracting the text and start time for each entry.
# 
# **Structure of Processed Text**
# 
# The processed text consists of lines formatted as: `Text: [spoken content] Start: [start time in seconds]`
# 
# We structure the processed text this way to provide a clear and concise representation of both the spoken content and its corresponding timestamp. This format is particularly useful for several reasons:
# 
# - **Readability:** The structure is simple and readable, making it easy to scan through the transcript and understand the flow of the content.
# 
# - **Timestamp retrieval:** By including the start time, users can easily locate the exact moment in the video where a specific segment of text appears. This is invaluable for tasks like video editing, content review, and detailed analysis.
# 
# - **Q&A applications:** In a Q&A system, having timestamps allows users to not only receive answers but also reference the specific part of the video for context. This enhances user experience by linking text answers to video content directly.
# 

# In[6]:


def process(transcript):
    # Initialize an empty string to accumulate processed text
    txt = ""

    # Iterate over each segment in the transcript
    for i in transcript:
        try:
            # Format the text and start time, then add to the accumulated string
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except:
            # If an error occurs (e.g., missing keys), skip the entry
            pass

    # Return the processed text
    return txt


# In[7]:


processed_transcript = process(transcript)
processed_transcript[:100] # Display the first 100 characters of the processed transcript


# ## Chunking the Transcript
# 
# Chunking is a crucial step in processing large text documents for use with embedding models and vector databases. It involves breaking down a long piece of text into smaller, manageable segments. This process is essential for several reasons:
# 
# - **Embedding model limitations**: Most embedding models have a maximum input length. Chunking ensures that each piece of text fits within these limits.
# 
# - **Granularity for retrieval**: Smaller chunks allow for more precise retrieval of relevant information. Instead of returning an entire document, a system can pinpoint specific sections that are most relevant to a query.
# 
# - **Computational efficiency**: Processing smaller chunks of text is often more computationally efficient than handling an entire document at once.
# 
# - **Context preservation**: By using techniques like overlapping chunks, we can maintain some context between adjacent segments of text.
# 
# This is the **second stage** of the RAG process, called **Splitting**, where the transcript is divided into smaller chunks to enable efficient embedding and precise retrieval of relevant information.
# 
# We use the `RecursiveCharacterTextSplitter` to break the processed transcript into smaller chunks, which are easier to handle during embedding and search.
# 

# ![RAG_Splitting.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/uuoPFzIpFzbQeQIcYh3EKA/RAG-Splitting.png)
# 

# In[8]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Maximum chunk size of 200 characters
    chunk_overlap=20  # Overlap of 20 characters between chunks
)

chunks = text_splitter.split_text(processed_transcript)
chunks[:10]  # Display the first 10 chunks


# ## Setting Up the watsonx Model
# 
# Now we set up the watsonx model for summarization and Q&A.
# 

# ### Credentials Setup
# 

# In this section, we set up the necessary credentials to access IBM Watson services and use the Granite model for our tasks. Specifically, we are utilizing the `Granite Generation 3 8B Instruct Model` model, which is a large-scale language model designed for conversational AI and other natural language processing tasks.
# 

# **Explanation of Parameters**
# 
# - **model_id**: Identifies the specific language model to be used. Here, it is set to Granite Generation 3 8B Instruct Model.
#   
# - **credentials**: Contains the authentication details necessary for accessing IBM Watson services. This typically includes the service URL and possibly an API key or token.
#   
# - **client**: An instance of `APIClient` that facilitates communication with the IBM Watson API using the provided credentials.
#   
# - **project_id**: A string identifier for the project, used to organize and manage tasks within the IBM Watson environment.
# 

# In[9]:


# Define the model ID for the Granite 8B Instruct Generation 3 model
model_id = "ibm/granite-3-8b-instruct"

# Set up the credentials needed to access the IBM Watson services
credentials = Credentials(
    url = "https://us-south.ml.cloud.ibm.com",
)

# Initialize the API client with the given credentials
client = APIClient(credentials)

# Define the project ID for organizing tasks within IBM Watson services
project_id = "skills-network"


# ### Defining Parameters for watsonx Model
# 

# In[10]:


parameters = {
    # Specifies the decoding method as greedy decoding
    # This means the model always chooses the most probable next token
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    
    # Sets the minimum number of new tokens to generate to 1
    # The model will always produce at least this many tokens
    GenParams.MIN_NEW_TOKENS: 1,
    
    # Sets the maximum number of new tokens to generate to 500
    # The model will stop generating after reaching this limit
    GenParams.MAX_NEW_TOKENS: 500,
    
    # Defines sequences that will cause the generation to stop
    # In this case, generation will stop when encountering two consecutive newlines
    GenParams.STOP_SEQUENCES: ["\n\n"],
}


# ### Initializing watsonx LLM
# 

# In[11]:


watsonx_granite = WatsonxLLM(
    # Specifies the ID of the model to be used
    # This is likely an enum or constant value defining a specific model
    model_id=model_id,
    
    # The URL endpoint for the Watson service
    # This is retrieved from a credentials dictionary
    url=credentials.get("url"),
    
    # The ID of the project in which this LLM instance will operate
    # This helps in organizing and managing different LLM instances
    project_id=project_id,
    
    # A dictionary of parameters that configure the behavior of the LLM
    # This includes settings like decoding method, token limits, and stop sequences
    params=parameters
)


# ## Embedding and Similarity Search
# 

# ### Embedding the Transcript Chunks
# 
# ![Vector Embedding-1.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/jbBnT-TYIwXw9cjF-eZFoA/Vector%20Embedding-1.png)
# 
# 
# **Embeddings** are dense vector representations of text that capture semantic meaning in a high-dimensional space. They allow for efficient comparison and similarity calculations between different pieces of text, making them crucial for tasks such as semantic search, content clustering, and information retrieval.
# 
# By converting the textual content of transcript chunks into these numerical representations, we create a powerful foundation for various natural language processing tasks. This enhances our ability to analyze and interact with the video content in sophisticated ways, enabling more accurate and efficient information retrieval and analysis.
# 
# To embed the transcript chunks, we are using the IBM SLATE-30M (ENG) model, part of IBM watsonX. This model is designed for English text and trained on 30 million parameters, generating rich semantic embeddings suitable for NLP tasks like similarity search and classification.
# 
# We will call `get_embedding_model_specs()` to retrieve information about available embedding models from the Watson service. This could be useful for understanding the capabilities and options available.
# 
# It creates an instance of WatsonxEmbeddings, which is likely a class that interfaces with IBM Watson's embedding models. This instance is configured to use the IBM SLATE 30M English model, with the specified service URL and project ID.
# 

# In[12]:


# Fetch specifications for available embedding models from the Watson service
get_embedding_model_specs(credentials.get('url'))

# Part 1: Create Embedding Model
# Set up the WatsonxEmbeddings object
embeddings = WatsonxEmbeddings(
    # Specifies the ID of the embedding model to be used
    # In this case, it's using the IBM SLATE 30M English model
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    
    # The URL endpoint for the Watson service
    # This is retrieved from the credentials dictionary
    url=credentials["url"],
    
    # The ID of the project in which this embedding model will operate
    # This helps in organizing and managing different model instances
    project_id=project_id
)


# ### Implementing FAISS for Similarity Search
# 
# `FAISS` (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors, making it ideal for large-scale data. To implement similarity search, we use `FAISS ` to find the most relevant transcript chunks based on a user query. 
# 
# We use the `from_texts` function, which converts the transcript chunks into embeddings and stores them in a **FAISS index**. This is the **storing stage** of the RAG process, where the chunked transcript embeddings are stored in the FAISS index to facilitate fast and efficient retrieval.
# 
# The resulting `faiss_index` can be used for fast similarity searches over the embedded text chunks, enabling efficient retrieval of relevant text passages based on semantic similarity.
# 

# ![RAG_Index_Storing.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/PGuLDODa54Gtz_hFHBji0Q/RAG-Index-Storing.png)
# 

# In[13]:


# Create a FAISS index from the text chunks using the specified embeddings
faiss_index = FAISS.from_texts(chunks, embeddings)


# ### Perform Similarity Search
# We can now search for specific queries within the embedded transcript using the FAISS similarity search to retrieve the most relevant chunks based on the user’s input.
# 

# In[14]:


# Define the query string we want to search for
query = "Which company they were talking about?"

# Perform a similarity search on the FAISS index
# The search returns the 'k' most similar chunks to the query
# In this case, k=3, so it returns the top 3 most similar results
results = faiss_index.similarity_search(query, k=3)

# Iterate through the results and print each one
for result in results:
    print(result)


# ### Explanation
# 
# In this example, we asked the question "Which company were they talking about?" The similarity search algorithm successfully retrieved relevant chunks of text from the transcript.
# 
# 1. **Identifying the Company**: The first result directly answers our query. It mentions "IBM" twice:
#    - "she's recently placed in IBM"
#    - "interview experience of IBM"
#    
#    This clearly indicates that the company being discussed is IBM.
# 
# 2. **How We Fetched It**:
#    - The FAISS index we created earlier contains vector representations (embeddings) of all the text chunks from the transcript.
#    - When we input our query, it's also converted into a vector using the same embedding model.
#    - The `similarity_search` function then finds the chunks whose vector representations are most similar to our query vector.
#    - We requested the top 3 most similar chunks (`k=3`), which are returned and printed.
# 
# 3. **Context**: The other results provide additional context about the interview process, mentioning details like:
#    - There were multiple interviewers, including a senior software developer.
#    - There was a technical round in the interview process.
# 
# This demonstrates the power of semantic search: even though our query didn't use the exact words found in the transcript, the system was able to understand the intent of the question and retrieve relevant information, successfully identifying IBM as the company being discussed.
# 

# ## Summarizing the Transcript with LLMChain
# 
# In this section, we will use `PromptTemplate` to define a prompt for summarizing the YouTube video transcript. We will then use `LLMChain` to generate a concise summary based on this prompt.
# 

# ### Define the Prompt Template
# The `PromptTemplate` helps structure the input to the language model by specifying the input variables and the template format for the summary. In this case, we want to summarize the transcript in a concise paragraph while ignoring any timestamps.
# 
# Following is a breakdown of its components and functionality:
# 
# **Components**
# 
# 1. **input_variables**:
#    - This is a list that defines the variables that will be used in the template.
#    - In our example, we have a single variable: `transcript`.
#    - This allows us to dynamically insert the actual transcript text when generating the prompt.
# 
# 2. **template**:
#    - This is a string that contains the format of the prompt.
#    - It includes both static text and placeholders for dynamic content (e.g., `{transcript}`).
#    - The template guides the AI on what kind of output is expected.
# 
# **Functionality**
# 
# - The `PromptTemplate` allows for:
#   - **Reusability**: Once defined, it can be used multiple times with different transcripts without rewriting the prompt structure.
#   - **Consistency**: Ensures that all prompts follow the same format, which can lead to more uniform responses from the AI.
#   - **Customization**: You can easily modify the template to change instructions or add additional context as needed.
# 

# In[15]:


# Define the prompt template for summarizing the transcript
prompt = PromptTemplate(
    # Specify the input variables that will be used in the template
    input_variables=["transcript"],
    
    # Define the actual template string
    template="""
Summarize the following YouTube video transcript in terms of paragraph:

{transcript}

Your summary should have concise summary in terms of paragraph. Ignore any timestamps.
"""
)


# ### Instantiate the LLMChain
# We will instantiate LLMChain with the defined prompt and our WatsonX language model. This chain will take the processed transcript and generate a summary.
# 

# In[16]:


# Instantiate LLMChain with the refined prompt and LLM
summarise_chain = LLMChain(
    llm=watsonx_granite,  # The language model to be used (in this case, watsonx_granite)
    prompt=prompt,        # The PromptTemplate we defined earlier
    verbose=True          # Enable verbose mode for detailed output
)

# Return the instantiated LLMChain object
summarise_chain


# ### Generate the Summary
# Now, we will pass the processed transcript to the LLMChain to get a summarized version.
# 

# In[17]:


# Pass the processed transcript to the LLMChain for summarization
summary = summarise_chain.predict(transcript=processed_transcript)

# Print the generated summary
print(summary)


# ## Retrieving Relevant Context and Generating Answers
# 
# In this section, we will define a function to retrieve relevant video context, create a prompt template for answering questions, and generate an answer using a language model.
# 

# ![Index Retrieval.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/cL-g5tcY9qsfwWCsrZkybg/Index%20Retrieval.png)
# 

# ### Explanation 
# 
# 1. **User query**: 
#    - A user asks a question about a specific YouTube video or a topic covered in multiple videos.
# 
# 2. **Query embedding**: 
#    - The user's question is converted into a vector embedding using the same model used for processing video transcripts (e.g., IBM SLATE-30M).
# 
# 3. **Similarity search**: 
#    - The query embedding is compared to the embeddings of transcript chunks stored in your vector database (FAISS index). 
#    - These transcript chunks represent segments of YouTube video.
# 
# 4. **Retrieval**: 
#    - The system retrieves the most similar transcript chunks based on the similarity scores. 
#    - These chunks are likely to contain information relevant to the user's question.
# 
# 5. **Context assembly**: 
#    - The retrieved transcript chunks are combined to form the relevant context. 
#    - This context might include portions of transcripts from the youtube video, depending on the query and available content.
# 

# ### Retrieving Relevant Context
# 
# We use the `retrieve` function to get relevant video context from a FAISS index based on the query provided. This function performs a similarity search to find the top 7 relevant documents.
# 

# In[18]:


def retrieve(query):
    # Perform a similarity search on the FAISS index
    relevant_context = faiss_index.similarity_search(query, k=7)
    return relevant_context


# ### Creating the Prompt Template
# The PromptTemplate is used to structure the prompt for the language model. It includes the relevant video context and the question that needs to be answered.
# 

# The `qa_template` is a carefully crafted prompt designed for a question-answering system based on video content. Let's break it down:
# 
# 1. **Role Definition**:
#    - **Text**: "You are an expert assistant providing detailed answers based on the following video content."
#    - **Purpose**: This sets the AI's role and expectations for its responses.
# 
# 2. **Context Placeholder**:
#    - **Text**: "Relevant Video Context: {context}"
#    - **Purpose**: This is where specific information from the video will be inserted.
# 
# 3. **Instruction**:
#    - **Text**: "Based on the above context, please answer the following question:"
#    - **Purpose**: This directs the AI to use the provided context for answering.
# 
# 4. **Question Placeholder**:
#    - **Text**: "Question: {question}"
#    - **Purpose**: This is where the user's specific question will be inserted.
# 

# In[19]:


# Define the template for question answering
qa_template = """
You are an expert assistant providing detailed answers based on the following video content.

Relevant Video Context: {context}

Based on the above context, please answer the following question:
Question: {question}
"""

# Create the PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["context", "question"],  # Variables to be filled dynamically
    template=qa_template
)


# ### Setting Up the LLMChain
# We instantiate LLMChain with the watsonx language model and the prompt template. This chain will generate answers based on the provided context and question.
# 

# In[20]:


# Create a Question-Answering Chain
QAChain = LLMChain(
    llm=watsonx_granite,     # The language model to be used (watsonx_granite in this case)
    prompt=prompt_template,  # The PromptTemplate we defined earlier for structuring the input
    verbose=True             # Enable verbose mode for detailed output
)

QAChain


# ### Generating an Answer
# To generate an answer, we first retrieve the relevant context using the retrieve function and then use QAChain to predict the answer based on this context and the given question.
# 

# In[21]:


def generate_answer(question):
    # Retrieve relevant context based on the question
    relevant_context = retrieve(question)
    
    # Use the QAChain to generate an answer based on the context and question
    answer = QAChain.predict(context=relevant_context, question=question)
    
    # Return the generated answer
    return answer


# In[23]:


# Input question (We can ask: Is there a coding round?)
question = input("Enter your question: ")

# Generate the answer
answer = generate_answer(question)
print(answer)


# ## Exercise
# 

# ### Exercise: 1 Ask More Questions
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# # Input question
# question = input("Enter your question: ")
# 
# # Generate the answer
# answer = generate_answer(question)
# print(answer)
# ```
# 
# </details>
# 

# ### Exercise: 2 Try a Different Video
# 

# ##### Set the Video ID: Assign a new video ID to the variable video_id.
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# video_id = "INSERT_VIDEO_ID_HERE"  # Replace with the actual video ID
# ```
# 
# </details>
# 

# ##### Retrieve the Transcript: Obtain the transcript for the specified video ID.
# 

# In[ ]:


#TODO 
transcript = get_transcript(osKyvYJ3PRM)


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# transcript = get_transcript(video_id)
# ```
# 
# </details>
# 

# ##### Process the Transcript: Clean and format the transcript.
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# processed_transcript = process(transcript)
# ```
# 
# </details>
# 

# ##### Chunk the Documents: Divide the processed transcript into smaller chunks for better indexing.
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# chunks = text_splitter.split_text(processed_transcript)
# ```
# 
# </details>
# 

# ##### Add Embeddings to FAISS: Incorporate the processed transcript embeddings into FAISS by overwriting the `faiss_index` variable, which will be used for retrieving the relevant context.
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# faiss_index = FAISS.from_texts(chunks, embeddings)
# ```
# 
# </details>
# 

# ##### Summarize the Transcript: Generate a summary of the processed transcript.
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# summary = summarise_chain.predict(transcript = processed_transcript)
# print(summary)
# ```
# 
# </details>
# 

# ##### Ask a Question: Prompt for a question about the video.
# 

# In[ ]:


#TODO


# <details>
#     <summary>Click here for hint</summary>
# 
# ```python
# # Input question
# question = input("Enter your question: ")
# 
# # Generate the answer
# answer = generate_answer(question)
# print(answer)
# ```
# 
# </details>
# 

# ## Conclusion
# 
# In this lab, you have learned how to:
# 
# - **Extract and process transcripts**: Retrieve and clean YouTube video transcripts for further analysis.
# - **Summarize video content**: Implement techniques to create concise summaries of video transcripts.
# - **Implement similarity search**: Use FAISS for similarity search to find relevant video segments.
# - **Create a Q&A system**: Develop a question-and-answer system using LangChain and RAG techniques.
# 
# This approach can be used to build intelligent systems capable of understanding and interacting with video content effectively.
# 
# ## Next Steps
# 
# 
# 1. **Explore other models and parameters**:
#     Investigate other language models and tuning parameters to improve summarization and question-answering performance.
# 
# 2. **Integrate into applications**:
#     Consider integrating this system into applications for automatic content summarization and question answering, enhancing user interaction with video content.
# 
# By following these next steps, you can further refine and expand your system's capabilities.
# 

# ## Authors
# 

# [Kunal Makwana](https://author.skills.network/instructors/kunal_makwana) is a Data Scientist at IBM and is currently pursuing his Master's in Computer Science at Dalhousie University.
# 

# ## Contributors
# 

# [Ricky Shi](https://author.skills.network/instructors/ricky_shi)
# 

# Copyright © 2024 IBM Corporation. All rights reserved.
# 
