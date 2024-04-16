# Custom-built LLM chatbot using llama-2 on CPU machine
Create your own custom-built Chatbot using the Llama 2 language model developed by Meta AI.
The possibilities with the Llama 2 language model are vast. Depending on your data set, you can train this model for a specific use case, such as **Customer Service and Support, Marketing and Sales, Human Resources, Legal Services, Hospitality, Insurance, Healthcare, Travel, and more**. 

It uses _Sentence Transformers for embeddings, FAISS CPU for vector storage, and the Chainlit library for a QA interface_. With RAG, the LLM will have access to **proprietary data** provided by you as custom data sources from which it can retrieve information. 
**With the use of Ctransfrmer, this can run smoothly on a regular CPU machine, so there is no need for expensive hardware** :smiley:

## Workflow:

1. A user makes a question to the LLM. Before reaching the model, the question reaches a retriever. 
2. This retriever will be responsible for finding relevant documents from the "knowledge base that you have provided" to answer the question. 
3. The question, plus the relevant documents, will then be sent to the LLM, which will be able to generate a source-informed answer according to the sources from the documents it received.

## Packages:

Before getting our hands dirty with code, let's look at some key packages needed as mentioned in ***requirements.txt***. 

 * [LangChain](https://www.langchain.com/): A framework that allows us to develop several applications powered by LLMs.
  
 * [LangChain](https://www.langchain.com/): A framework that allows us to develop several applications powered by LLMs.
  
 * [Sentence Transformers](https://pypi.org/project/sentence-transformers/): A framework that provides an easy method to compute dense vector representations for sentences, paragraphs, and images by leveraging pre-trained transformer models. We are using all mini LLM v6.
  
 * [LangChain](https://www.langchain.com/): A framework that allows us to develop several applications powered by LLMs.
  
 * [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/) - Facebook AI Similarity Search (Faiss) is a library for efficient similarity search and clustering of dense vectors. It allows users to index vectors and search for the most similar vectors within the index
  
 * [chainlit](https://docs.chainlit.io/get-started/overview) - Chainlit is an open-source Python package to build production ready Conversational AI or agentic applications.
  
 * [Llama 2 Model](https://huggingface.co/meta-llama) (Quantized one): [llama-2-7b-chat.Q8_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf)

 * [Ctransformers](https://github.com/marella/ctransformers): Python bindings for the Transformer models implemented in C/C++

 * PyPDF2 (for PDF document loading)

## Steps to build the model:

### Pre-requisite: 

- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/) OR use the Python command to create a virtual environment. Ensure Python 3.6 or higher!
- Install [Visual Studio code](https://code.visualstudio.com/download) - or you’re your favorite editor
- Copy your knowledge base in the Data directory and in pdf format.
- Make sure Python and Chainlit is on the system path
- Download **llama-2-7b-chat.Q8_0.gguf** from (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main) in Llama2 directory
	
### Create a virtual environment & activate it.
```
	 conda create --name <env_name> [--prefix <env_path>] OR python -m venv <env_name>
	 conda activate <env_name> OR source <env_name>/bin/activate	# On win use <env_name>\Scripts\activate\
```
### Build the model
```
	 pip install -r requirements.txt            # Install all the packages mentioned above
	 python ingest.py                           # This will take 4-5 min or depending on the data size
	 chainlit run model.py -w                   # This will invoke localhost:8000 page, and your customer-built LLM is up :+1
```
Ask your queries.. they can be anything within the scope of your documents. You will get an answer and the source of the answer. Depending on your system configuration, the response to the query will take some time. For me, it took 10-15 seconds on 16 GB RAM.

Port and hostname are configurable through the CHAINLIT_HOST and CHAINLIT_PORT env variables. You can also use --host and --port when running chainlit run!

## License
This project is licensed under the MIT License.
