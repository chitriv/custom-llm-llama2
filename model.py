# Import the lib that we need
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

# Provide path of our embeddings
DB_FAISS_PATH = "vectorestores/db_faiss"

# The better the prompt is the better will be the response
customer_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know the answer.
Don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else
Helpful answer:
"""
# Setting custom prompt
def set_custom_prompt():
    """
    Prompt template for QA retrival for each vecorstores
    """

    prompt = PromptTemplate(template=customer_prompt_template, input_variables=['context','question'])

    return prompt

# retirval QA chain
def retrival_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={'k' : 2}),
        return_source_documents = True,                     # return only what has mentioned in the document and not the LLM's native training data
        chain_type_kwargs = {'prompt': prompt}
    )
    return qa_chain

# Load the LLM
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGUF",     # Provide the name of the hugging face LLM model: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
        model_type = "llama",
        max_new_token = 512,
        temprature = 0.5
    )
    return llm


# QA MOdel function    
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device' : 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    #db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrival_qa_chain(llm, qa_prompt, db)

    return qa

# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})       # Name - Value pair
    return response


## Chainlit: Provides conversational interface like FLASK,  ####
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the Custom-built Bot using RAG, FAISS....")
    await msg.send()
    msg.content = "Hi, Welcome to Custom-built Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)
    # explore if you want to add images

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True
    # The function `acall` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use ainvoke instead.
    # res = await chain.acall(message.content, callbacks=[cb])
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n Sources: " + str(sources)
    else:
        answer += f"\n No sources found"

    await cl.Message(content=answer).send()



"""
@cl.on_chat_start
async def start():
    image = cl.Image(path="./cat.jpeg", name="image1", display="inline")

    # Attach the image to the message
    await cl.Message(
        content="This message has an image!",
        elements=[image],
    ).send()
"""