import os
import time
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS   
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class KnowledgeBaseTool:
    def __init__(self, faiss_index_path="faiss_index", qwen_model_name="Qwen/Qwen3-4B-Instruct-2507"):
        """
        Инициализация всех компонентов RAG: FAISS, BM25 и Qwen для Query Expansion.
        """
        print("🚀 Инициализация KnowledgeBaseTool...")
        
        # Определяем устройство (XPU для Intel ARC)
        self.device = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu"
        print(f"📡 Используемое устройство: {self.device}")

        # 1. Загрузка эмбеддингов и FAISS (Dense Search)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": self.device}
        )
        self.vectorstore = FAISS.load_local(faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
        
        # 2. Подготовка BM25 (Sparse Search)
        print("📚 Подготовка BM25 индекса...")
        self.all_docs = list(self.vectorstore.docstore._dict.values())
        self.all_texts = [doc.page_content for doc in self.all_docs]
        
        # Настройка NLTK
        try:
            self.stop_words = set(stopwords.words('russian'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('russian'))

        tokenized_corpus = [self._preprocess(text) for text in self.all_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Загрузка Qwen для генерации подзапросов
        print(f"🧠 Загрузка Qwen ({qwen_model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        if self.device == "xpu":
            self.qwen_model = self.qwen_model.to("xpu")
            self.qwen_model = ipex.optimize(self.qwen_model, dtype=torch.bfloat16)

        self.qwen_pipe = pipeline(
            "text-generation",
            model=self.qwen_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device=0 if self.device == "xpu" else -1,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=True,
        )
        print("✅ KnowledgeBaseTool успешно инициализирован.")

    def _preprocess(self, text):
        """Очистка текста для BM25 токенизации."""
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t not in string.punctuation and t not in self.stop_words]

    def _extract_queries(self, generated_text):
        """Парсинг ответа Qwen для извлечения чистых запросов."""
        if "<|im_start|>assistant" in generated_text:
            assistant_part = generated_text.split("<|im_start|>assistant")[-1]
        else:
            assistant_part = generated_text
        
        content = assistant_part.split("<|im_end|>")[0].strip()
        return [line.strip() for line in content.split('\n') if line.strip()]

    def search(self, user_query: str, k_per_query: int = 3) -> str:
        """
        Основной метод поиска:
        1. Генерирует 3-5 подзапросов через Qwen.
        2. Проводит гибридный поиск по каждому.
        3. Дедуплицирует и возвращает контекст.
        """
        print(f"🔎 Инструмент поиска получил запрос: '{user_query}'")
        
        # Шаг 1: Генерация подзапросов
        prompt = f"<|im_start|>system\nСгенерируй 3 кратких поисковых запроса. Только запросы, без текста.<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        outputs = self.qwen_pipe(prompt)
        sub_queries = self._extract_queries(outputs[0]["generated_text"])
        if not sub_queries:
            sub_queries = [user_query]

        # Шаг 2: Гибридный поиск
        context_fragments = []
        seen_content = set()

        for q in sub_queries:
            # Dense (FAISS)
            dense_results = self.vectorstore.similarity_search(q, k=k_per_query)
            # Sparse (BM25)
            tokenized_q = self._preprocess(q)
            bm25_scores = self.bm25.get_scores(tokenized_q)
            top_indices = bm25_scores.argsort()[-k_per_query:][::-1]
            sparse_results = [self.all_docs[i] for i in top_indices]

            for doc in (dense_results + sparse_results):
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    context_fragments.append(doc.page_content)

        return "\n\n---\n\n".join(context_fragments[:10]) # Ограничиваем объем для агента