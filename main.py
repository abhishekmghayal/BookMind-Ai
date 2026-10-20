from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()
embeddings_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings_model
)

retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":4,
                   "fetch_k":10,
                   "lambda_mult":0.5
                   },
                )

llm=ChatGroq(model="qwen/qwen3.8-27b")

multi_query_retiever=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
    )




#prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",""" you are a helpful ai assistant 
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context, 
            say: 'I could not find the answer in the document."""),
            ("human",""" Context: {context} Question: {question}""")
    ]
)

print("Rag System Creates @@@@@@@@@@@@@")
print("press 0 to exit 0 &&&&&&")



while True:
    query=input("You:   ")
    if query=="0":
        break
    docs=multi_query_retiever.invoke(query)
    #docs=retriever.invoke(query)

    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context, 
        "question":query
        })

    response=llm.invoke(final_prompt)
    
    print("=========================>>>>>>>>>>>>>>")
    print(f"\n AI : {response.content}")
    print("=========================>>>>>>>>>>>>>>")
