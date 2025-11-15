import intel_extension_for_pytorch as ipex
from fastapi import FastAPI, HTTPException
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


print("Загрузка FAISS индекса...")
device = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu"
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
if device == "xpu":
    qwen_model = qwen_model.to("xpu")
    qwen_model = ipex.optimize(qwen_model, dtype=torch.bfloat16)
qwen_pipe = pipeline(
    "text-generation",
    model=qwen_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=0 if device == "xpu" else -1,  
    max_new_tokens=128,
    temperature=0.2,
    do_sample=True,
)
gost_query_template = "{% if not messages[0]['role'] == 'system' %}<|im_start|>system\n Определи какие ключевые моменты нужно узнать. Сгенерируй ТОЛЬКО  от 3 до 8 поисковых запроса. Формат: каждый запрос с новой строки, без пояснений. Генерируй чётко и кратко Пример: Требования к сварным швам<|im_end|>\n{% endif %}<|im_start|>user\n{{ messages[-1]['content'] }}<|im_end|>\n<|im_start|>assistant\n"
tokenizer.chat_template = gost_query_template
app = FastAPI(title="RAG API")
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

def answer_question(user_query: str) -> str:
    print(f"\n🔍 Начало обработки запроса: '{user_query}'", flush=True)
    start_time = time.time()

    # === Этап 1: Генерация поисковых запросов ===
    print("⚙️ Этап 1/4: Генерация поисковых запросов через Qwen...", flush=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_query}],
        tokenize=False,
        add_generation_prompt=True
    )
    gen_start = time.time()
    outputs = qwen_pipe(prompt)
    gen_time = time.time() - gen_start
    raw_output = outputs[0]["generated_text"]
    search_queries = extract_search_queries(raw_output)
    if not search_queries:
        search_queries = [user_query]
    print(f"✅ Сгенерировано {len(search_queries)} запрос(ов): {search_queries} (за {gen_time:.1f}с)", flush=True)
    
    # === Этап 2: Hybrid Dense + Sparse поиск по каждому сгенерированному запросу ===
    print("⚙️ Этап 2/4: Hybrid Dense + Sparse поиск...", flush=True)
    hybrid_docs = []
    for q in search_queries:
        print(f"  Обработка подзапроса: '{q}'", flush=True)
        
        # Dense: FAISS
        dense_results = vectorstore.similarity_search(q, k=5)
        # Sparse: BM25
        tokenized_q = preprocess(q)
        bm25_scores = bm25.get_scores(tokenized_q)
        top_n_sparse = 5
        top_indices = bm25_scores.argsort()[-top_n_sparse:][::-1]
        sparse_results = [all_docs[i] for i in top_indices]
        # Объединяем и дедуплицируем
        combined = dense_results + sparse_results
        seen = set()
        unique_for_query = []
        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_for_query.append(doc)
        hybrid_docs.extend(unique_for_query)
    # Глобальная дедупликация по всему результату
    seen_global = set()
    final_docs = []
    for doc in hybrid_docs:
        if doc.page_content not in seen_global:
            seen_global.add(doc.page_content)
            final_docs.append(doc)
    print(f"✅ Hybrid поиск: {len(final_docs)} уникальных фрагментов", flush=True)
    query_context_pairs = []
    for doc in final_docs:
        truncated = doc.page_content[:1000]
        query_context_pairs.append(("", truncated))


    # === Этап 3: Формирование промпта ===
    print("⚙️ Этап 3/4: Формирование финального промпта для Mistral...", flush=True)
    prompt_parts = [
        "Ты помощник, отвечающий строго по контексту.",
        "Если части ответов не найдено — скажи 'Точный ответ на данный вопрос не найден в предоставленных документах'.\n"
    ]
    for i, (_, ctx) in enumerate(query_context_pairs, 1):
        prompt_parts.append(f"Документ {i}:\n[Фрагмент]\n{ctx}\n")
    prompt_parts.append(f"На основе всего приведённого контекста дай один чёткий и полный ответ на исходный вопрос пользователя: '{user_query}'. Не перечисляй ответы на промежуточные запросы.")
    full_instruction = "\n".join(prompt_parts)
    print("✅ Промпт для Mistral сформирован", flush=True)

    # === Этап 4: Вызов Mistral ===
    print("⚙️ Этап 4/4: Отправка запроса в Mistral API...", flush=True)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        error_msg = "❌ Ошибка: MISTRAL_API_KEY не установлен"
        print(error_msg, flush=True)
        return error_msg

    client = Mistral(api_key=api_key)
    try:
        mistral_start = time.time()
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": full_instruction}]
        )
        answer = chat_response.choices[0].message.content
        mistral_time = time.time() - mistral_start
        total_time = time.time() - start_time
        print(f"✅ Ответ получен от Mistral (за {mistral_time:.1f}с)", flush=True)
        print(f"⏱️ Общее время обработки: {total_time:.1f} секунд", flush=True)
        return answer
    except Exception as e:
        error_msg = f"❌ Ошибка при вызове Mistral: {str(e)}"
        print(error_msg, flush=True)
def answer_question_without_rag(user_query: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Ошибка: MISTRAL_API_KEY не установлен"
    client = Mistral(api_key=api_key)
    try:
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": user_query}]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Ошибка: {str(e)}"       
class QuestionRequest(BaseModel):
    question: str
    use_rag: bool = True  # по умолчанию — с RAG

@app.post("/ask")
async def ask(request: QuestionRequest):
    try:
        if request.use_rag:
            answer = answer_question(request.question)  # ← RAG-версия
        else:
            answer = answer_question_without_rag(request.question)  # ← прямой Mistral
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))