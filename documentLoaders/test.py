from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

splitter=CharacterTextSplitter(chunk_size=10,chunk_overlap=1,separator="")

data=TextLoader("documentLoaders/notes.txt")



docs=data.load()
chunk=splitter.split_documents(docs)

for i in chunk:
    print(i.page_content)
    print("--------")


print(len(chunk))
#print(docs[0].page_content)


