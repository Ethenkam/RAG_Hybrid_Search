import os
import re
import unicodedata
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_text(text):
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'Страниц[\u0430у].*\d+.*', '', text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    return text

def load_documents(directory):
    print(f"🔍 Ищу .txt файлы в: {os.path.abspath(directory)}")
    all_docs = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                print(f"\n--- Обработка файла: {file_path} ---")
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    raw_docs = loader.load()
                    if not raw_docs:
                        continue
                    cleaned_docs = []
                    for doc in raw_docs:
                        cleaned_content = clean_text(doc.page_content)
                        if cleaned_content:
                            cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                    if cleaned_docs:
                        separators = ["\n\n", "\n", " ", ". ", ""]
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            separators=separators
                        )
                        docs = text_splitter.split_documents(cleaned_docs)
                        all_docs.extend(docs)
                except UnicodeDecodeError:
                    try:
                        loader = TextLoader(file_path, encoding="cp1251")
                        raw_docs = loader.load()
                        cleaned_docs = []
                        for doc in raw_docs:
                            cleaned_content = clean_text(doc.page_content)
                            if cleaned_content:
                                cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                        if cleaned_docs:
                            separators = ["\n\n", "\n", " ", ". ", ""]
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=300,
                                chunk_overlap=50,
                                separators=separators
                            )
                            docs = text_splitter.split_documents(cleaned_docs)
                            all_docs.extend(docs)
                    except Exception as e2:
                        print(f"Ошибка cp1251 в {file_path}: {e2}")
                except Exception as e:
                    print(f"Ошибка в {file_path}: {e}")
    return all_docs