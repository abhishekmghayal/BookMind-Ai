from langchain_community.document_loaders import WebBaseLoader
url="https://www.apple.com/in/macbook-pro/"

data=WebBaseLoader(url)
docs=data.load()
print(docs)