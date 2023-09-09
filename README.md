**How to Chat with Your PDF using Python & Llama2**
With the recent release of Meta’s Large Language Model(LLM) Llama-2, the possibilities seem endless.

What if you could chat with a document, extracting answers and insights in real-time?

Well with Llama2, you can have your own chatbot that engages in conversations, understands your queries/questions, and responds with accurate information.

In this article, we’ll reveal how to create your very own chatbot using Python and Meta’s Llama2 model.

If you want help doing this, you can schedule a FREE call with us at www.woyera.com where we can guide you through the process to make a chatbot tailored to your needs.

**What You Will Need:**

  **Pinecone API Key**: The Pinecone vector database can store vector embeddings of documents or conversation history, allowing the chatbot to retrieve relevant responses based on the user’s input.
  
  **Streamlit Replicate API Key**: This is how we will apply the Llama2 model for our chatbot.
  
**Python**: The programming language we will be using to build our chatbot. (Make sure to download Python versions 3.8 or higher)

**Steps for Pinecone:**
    
    1) Sign up for an account on the Pinecone website.
    
    2) Once you are signed up and logged in, on the left side navigation menu click “API Keys”.

    3) Copy the API key displayed on the screen (we will use this key later).

    4) Now, go back to the “Indexes” tab and create a new index.
    
    5) Name it whatever you want and make the dimensions 768.
    
    6) Create the index and copy the environment of the index, we’ll need it for later.


**Steps for how to get Replicate API Key:**

    1) Go to the Replicate website and sign up.
    
    2) Once you are signed up and logged in, navigate to this link to see your API Key: https://replicate.com/account/api-tokens
    
    3) You should see your own key on the page. Copy the key and save it for later.

**Lets start Building our Chatbot!**

    Before diving into the code, ensure that you have the required libraries installed. You can install them using the following commands into your terminal:


    pip install pinecone-client langchain

**Step 1: Initializing the Environment
**

    Make a python file ex. app.py and open it with your code editing application of choice.


    Next, we need to set up the environment with the necessary libraries and tokens. The code snippet below imports the libraries and initializes the Replicate API and     Pinecone, enabling us to access their functionalities. Paste your previously copied API keys in the designated spots.


    import os
    
    import sys
    
    import pinecone
    
    from langchain.llms import Replicate
    
    from langchain.vectorstores import Pinecone
    
    from langchain.text_splitter import CharacterTextSplitter
    
    from langchain.document_loaders import PyPDFLoader
    
    from langchain.embeddings import HuggingFaceEmbeddings
    
    from langchain.chains import ConversationalRetrievalChain
    
    
    # Replicate API token
    
    os.environ['REPLICATE_API_TOKEN'] = "YOUR REPLICATE API HERE"
    
    
    # Initialize Pinecone
    
    pinecone.init(api_key='YOUR PINECONE API HERE', environment='YOUR ENVIRONMENT HERE')



**Step 2: Preparing the Data
**

    Next, we need data to build our chatbot.
    
    
    In this example, we load a PDF document in the same directory as the python application and prepare it for processing by splitting it into smaller chunks using the CharacterTextSplitter.


    # Load and preprocess the PDF document
    
    loader = PyPDFLoader('./NAME_OF_YOUR_PDF_HERE.pdf')
    
    documents = loader.load()
    
    
    # Split the documents into smaller chunks for processing
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    texts = text_splitter.split_documents(documents)


**Step 3: Leveraging HuggingFace Embeddings
**

    HuggingFace’s embeddings play a crucial role in transforming the text data into dense numerical vectors, a fundamental step in building our intelligent chatbot.

    # Use HuggingFace embeddings for transforming text into numerical vectors
    
    embeddings = HuggingFaceEmbeddings()


**Step 4: Creating the Pinecone Vector Database**

    With our text data converted into embeddings, we can proceed to create the Pinecone vector database.

    This database allows for efficient similarity search and retrieval of vectors, which is essential for our chatbot’s conversational capabilities.

    # Set up the Pinecone vector database
    
    index_name = "NAME OF YOUR DATABASE(PINECONE INDEX THAT YOU CREATED)"
    
    index = pinecone.Index(index_name)
    
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)


**Step 5: Integrating Replicate LLama2 Model**

    Llama2, with its powerful language generation model, is the heart of our chatbot.

    We integrate this model to enable our chatbot to understand user queries and generate intelligent responses.



    # Initialize Replicate Llama2 Model
    
    llm = Replicate(
    
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        
        input={"temperature": 0.75, "max_length": 3000}
        
    )


**Step 6: Building the Conversational Retrieval Chain**

To create a dynamic and interactive chatbot, we construct the ConversationalRetrievalChain by combining Llama2 LLM and the Pinecone vector database.

This chain enables the chatbot to retrieve relevant responses based on user queries and the chat history.

    # Set up the Conversational Retrieval Chain
    
    qa_chain = ConversationalRetrievalChain.from_llm(
    
        llm,
        
        vectordb.as_retriever(search_kwargs={'k': 2}),
        
        return_source_documents=True
        
    )


**Step 7: Chatting with the Chatbot**

The code snippet below initiates an infinite loop, where the user inputs queries and the chatbot responds with intelligent answers based on the chat history.



    # Start chatting with the chatbot
    
    chat_history = []
    
    while True:
    
        query = input('Prompt: ')
        
        if query.lower() in ["exit", "quit", "q"]:
        
            print('Exiting')
            
            sys.exit()
            
        result = qa_chain({'question': query, 'chat_history': chat_history})
        
        print('Answer: ' + result['answer'] + '\n')
        
        chat_history.append((query, result['answer']))

**Final Step: Test the Chatbot!**

    By this point, all of your code should be put together and you should now be able to chat with your PDF document.
    
    Now simply run the command to execute the python application into your terminal ex. python app.py
    Test the chatbot!


**Congratulations!
**   
    You’ve now built an intelligent chatbot for your documents using the Llama2 model. Have fun!

