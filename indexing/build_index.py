import sys
import os
from tqdm import tqdm
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from data_loader.load_and_clean import load_documents
EMBEDDING_BATCH_SIZE = 16  # уменьшите до 8, если возникает нехватка памяти на GPU
def build_and_save_index(data_dir: str, save_path: str = "api\faiss_index"):
    
    docs = load_documents(data_dir)
    print(f"Загружено {len(docs)} чанков")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)             
    final_save_path = os.path.join(project_root, save_path)
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    print(f"Используемое устройство для эмбеддингов: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    if device == "xpu":
        embeddings.client = embeddings.client.to("xpu")
        print("✅ Эмбеддинг-модель перемещена на XPU")
    texts = [doc.page_content for doc in docs]
    embeddings_list = []
    batch_size = EMBEDDING_BATCH_SIZE
    for i in tqdm(range(0, len(texts), batch_size), desc="Генерация эмбеддингов"):
        batch = texts[i:i+batch_size]
        embs = embeddings.embed_documents(batch)
        embeddings_list.extend(embs)
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings_list)),
        embedding=embeddings,
        metadatas=[doc.metadata for doc in docs]
    )
    
    vectorstore.save_local(final_save_path)
    print(f"Индекс сохранён в: {final_save_path}")
if __name__ == "__main__":
        build_and_save_index("documents", r"faiss_index")
