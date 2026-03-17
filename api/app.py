from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os
import time
from mistralai import Mistral
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from rank_bm25 import BM25Okapi

# Определение устройства: CUDA > XPU > CPU
print("Загрузка FAISS индекса...")
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"
print(f"Используемое устройство: {device}")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": device}
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# === Инициализация BM25 для sparse retrieval ===
print("Подготовка BM25 индекса...")
try:
    all_docs = list(vectorstore.docstore._dict.values())
    all_texts = [doc.page_content for doc in all_docs]
    print(f"✅ Извлечено {len(all_texts)} документов для BM25")
except Exception as e:
    raise RuntimeError("Не удалось загрузить документы из FAISS. Убедитесь, что индекс сохранён с docstore.")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for res in ['punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{res}' if res == 'punkt_tab' else f'corpora/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

stop_words = set(stopwords.words('russian'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return tokens

print("Токенизация корпуса для BM25...")
tokenized_corpus = [preprocess(text) for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)
print("✅ BM25 готов")
qwen_model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
if device in ("cuda", "xpu"):
    qwen_model = qwen_model.to(device)
    if device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            qwen_model = ipex.optimize(qwen_model, dtype=torch.bfloat16)
        except ImportError:
            pass
qwen_pipe = pipeline(
    "text-generation",
    model=qwen_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=0 if device in ("cuda", "xpu") else -1,
    max_new_tokens=128,
    temperature=0.2,
    do_sample=True,
)
gost_query_template = "{% if not messages[0]['role'] == 'system' %}<|im_start|>system\n Определи какие ключевые моменты нужно узнать. Сгенерируй ТОЛЬКО  от 3 до 8 поисковых запроса. Формат: каждый запрос с новой строки, без пояснений. Генерируй чётко и кратко Пример: Требования к сварным швам<|im_end|>\n{% endif %}<|im_start|>user\n{{ messages[-1]['content'] }}<|im_end|>\n<|im_start|>assistant\n"
tokenizer.chat_template = gost_query_template
app = FastAPI(title="RAG API")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. API is running at /docs"}

@app.get("/health")
async def health():
    return {"status": "ok", "device": device}
def extract_search_queries(generated_text):
  
    if "<|im_start|>assistant" in generated_text:
        assistant_part = generated_text.split("<|im_start|>assistant")[-1]
    else:
        assistant_part = generated_text

    if "<|im_end|>" in assistant_part:
        assistant_part = assistant_part.split("<|im_end|>")[0]


    assistant_part = assistant_part.strip()

    if "</think>" in assistant_part:

        queries_part = assistant_part.split("</think>")[-1].strip()
    else:

        queries_part = assistant_part


    search_queries = [line.strip() for line in queries_part.split('\n') if line.strip()]
    
    return search_queries

def answer_question(user_query: str) -> dict:
    print(f"\n🔍 Начало обработки запроса: '{user_query}'", flush=True)
    start_time = time.time()
    timings = {}

    # === Этап 1: Генерация поисковых запросов ===
    print("⚙️ Этап 1/4: Генерация поисковых запросов через Qwen...", flush=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_query}],
        tokenize=False,
        add_generation_prompt=True
    )
    gen_start = time.time()
    outputs = qwen_pipe(prompt)
    timings["query_expansion"] = round(time.time() - gen_start, 2)
    raw_output = outputs[0]["generated_text"]
    search_queries = extract_search_queries(raw_output)
    if not search_queries:
        search_queries = [user_query]
    print(f"✅ Сгенерировано {len(search_queries)} запрос(ов): {search_queries} (за {timings['query_expansion']}с)", flush=True)

    # === Этап 2: Hybrid Dense + Sparse поиск ===
    print("⚙️ Этап 2/4: Hybrid Dense + Sparse поиск...", flush=True)
    search_start = time.time()
    hybrid_docs = []
    for q in search_queries:
        dense_results = vectorstore.similarity_search(q, k=5)
        tokenized_q = preprocess(q)
        bm25_scores = bm25.get_scores(tokenized_q)
        top_indices = bm25_scores.argsort()[-5:][::-1]
        sparse_results = [all_docs[i] for i in top_indices]
        combined = dense_results + sparse_results
        seen = set()
        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                hybrid_docs.append(doc)
    seen_global = set()
    final_docs = []
    for doc in hybrid_docs:
        if doc.page_content not in seen_global:
            seen_global.add(doc.page_content)
            final_docs.append(doc)
    timings["hybrid_search"] = round(time.time() - search_start, 2)
    print(f"✅ Hybrid поиск: {len(final_docs)} уникальных фрагментов", flush=True)

    sources = []
    query_context_pairs = []
    for doc in final_docs:
        truncated = doc.page_content[:1000]
        query_context_pairs.append(("", truncated))
        sources.append({
            "content": truncated[:200] + "..." if len(truncated) > 200 else truncated,
            "metadata": doc.metadata if doc.metadata else {}
        })

    # === Этап 3: Формирование промпта ===
    prompt_parts = [
        "Ты помощник, отвечающий строго по контексту.",
        "Если части ответов не найдено — скажи 'Точный ответ на данный вопрос не найден в предоставленных документах'.\n"
    ]
    for i, (_, ctx) in enumerate(query_context_pairs, 1):
        prompt_parts.append(f"Документ {i}:\n[Фрагмент]\n{ctx}\n")
    prompt_parts.append(f"На основе всего приведённого контекста дай один чёткий и полный ответ на исходный вопрос пользователя: '{user_query}'. Не перечисляй ответы на промежуточные запросы.")
    full_instruction = "\n".join(prompt_parts)

    # === Этап 4: Вызов Mistral ===
    print("⚙️ Этап 4/4: Отправка запроса в Mistral API...", flush=True)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return {"answer": "Ошибка: MISTRAL_API_KEY не установлен", "error": True}

    client = Mistral(api_key=api_key)
    try:
        mistral_start = time.time()
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": full_instruction}]
        )
        answer = chat_response.choices[0].message.content
        timings["mistral"] = round(time.time() - mistral_start, 2)
        timings["total"] = round(time.time() - start_time, 2)
        print(f"✅ Ответ получен от Mistral (за {timings['mistral']}с)", flush=True)
        print(f"⏱️ Общее время обработки: {timings['total']} секунд", flush=True)
        return {
            "answer": answer,
            "search_queries": search_queries,
            "num_sources": len(final_docs),
            "sources": sources[:10],
            "timings": timings,
            "device": device
        }
    except Exception as e:
        return {"answer": f"Ошибка при вызове Mistral: {str(e)}", "error": True}
def answer_question_without_rag(user_query: str) -> dict:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return {"answer": "Ошибка: MISTRAL_API_KEY не установлен", "error": True}
    client = Mistral(api_key=api_key)
    try:
        start_time = time.time()
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": user_query}]
        )
        total_time = round(time.time() - start_time, 2)
        return {
            "answer": chat_response.choices[0].message.content,
            "timings": {"mistral": total_time, "total": total_time},
            "device": device
        }
    except Exception as e:
        return {"answer": f"Ошибка: {str(e)}", "error": True}

class QuestionRequest(BaseModel):
    question: str
    use_rag: bool = True

@app.post("/ask")
async def ask(request: QuestionRequest):
    try:
        if request.use_rag:
            result = answer_question(request.question)
        else:
            result = answer_question_without_rag(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))