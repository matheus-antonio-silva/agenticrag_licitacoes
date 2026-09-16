from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    openai_api_key : str

    vector_store_path : str = "data/vectorstore"

    collection_name : str = "licitacoes"

    embedding_model : str = "text-embedding-3-small"

    llm_model : str  = 'gpt-5.6-luna'

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8'
    )

settings = Settings()