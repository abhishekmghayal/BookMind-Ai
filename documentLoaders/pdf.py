from langchain_community.document_loaders import PyPDFLoader
#from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
data=PyPDFLoader("documentLoaders/GRU.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10

)

#splitter=TokenTextSplitter(
 #   chunk_size=1000,
 #   chunk_overlap=10,
#)
chunk=splitter.split_documents(docs)
print(chunk[0].page_content)
