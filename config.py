#API配置
OPENAI_API_KEY = "sk-253f5b4a11b347a6892bff90f2b017b9"
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-max"
OPENAI_EMBEDDING_MODEL = "text-embedding-v3"

# 数据目录配置
DATA_DIR = "./data"

#向量数据库配置
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "course_materials"

# 文本处理配置
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
MAX_TOKENS = 4096

# RAG配置
TOP_K = 10
