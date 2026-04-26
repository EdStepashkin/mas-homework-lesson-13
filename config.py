import os
from datetime import datetime
from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Завантажуємо змінні до середовища, щоб OpenAI та інші бібліотеки їх бачили
load_dotenv()

class Settings(BaseSettings):
    # Явно вказуємо, що цю змінну треба брати з GEMINI_API_KEY у файлі .env
    api_key: SecretStr = Field(alias="GEMINI_API_KEY")

    # Задаємо дефолтну модель, щоб не тягнути її з .env
    model_name: str = "gemini-3-flash-preview"

    max_search_results: int = 5
    max_url_content_length: int = 5000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3

    # Content Pipeline
    output_dir: str = "example_output"
    max_revisions: int = 5
    max_iterations: int = 30

    # Langfuse
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str = Field(default="https://us.cloud.langfuse.com", alias="LANGFUSE_BASE_URL")

    # Правильний синтаксис конфігурації для Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Ігнорує зайві змінні в .env, якщо вони там є
    )

# Створюємо глобальний об'єкт (синглтон), який будемо імпортувати в інші файли
settings = Settings()


# ─────────────────────────────────────────────
# Langfuse client initialization (singleton)
# ─────────────────────────────────────────────
from langfuse import Langfuse, get_client

# Ініціалізуємо Langfuse singleton — всі інші модулі використовують get_client()
_langfuse_client = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_base_url,
)


# ─────────────────────────────────────────────
# Shared LLM instance for all agents
# ─────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=settings.model_name,
    api_key=settings.api_key.get_secret_value(),
    temperature=0.2,
)


# ─────────────────────────────────────────────
# Prompt Management: завантаження з Langfuse
# ─────────────────────────────────────────────
def get_prompt(name: str) -> str:
    """
    Завантажує system prompt з Langfuse Prompt Management за іменем.
    Використовує label 'production'. Компілює з поточною датою.
    
    Args:
        name: Ім'я промпту в Langfuse (e.g. "content-strategist")
    
    Returns:
        Скомпільований текст промпту з підставленими змінними.
    """
    langfuse = get_client()
    prompt = langfuse.get_prompt(name, label="production")
    compiled = prompt.compile(current_date=datetime.now().strftime("%Y-%m-%d"))
    return compiled