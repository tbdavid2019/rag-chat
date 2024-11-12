import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma  # 更新导入
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()


# 获取环境变量或使用默认值
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("未找到 OPENAI_API_KEY，请在 .env 文件中设置。")

OPENAI_API_BASE = os.getenv(
    "OPENAI_API_BASE",
    "http://Bedroc-Proxy-RCrbdJs2OHgQ-1123811820.us-west-2.elb.amazonaws.com/api/v1"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "cohere.embed-multilingual-v3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
WATCH_FOLDER = os.getenv('PDF_FOLDER', '/content/data')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', './chroma_db')

# 设置 OpenAI API
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

# 初始化嵌入模型
embedding_function = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)

# 初始化向量数据库
vectorstore = Chroma(
    collection_name="pdf_embeddings",
    embedding_function=embedding_function,
    persist_directory=CHROMA_DB_DIR
)



class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            process_pdf(event.src_path)

def process_pdf(pdf_path):
    print(f"Processing {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)

    # 添加来源信息
    for doc in docs:
        doc.metadata['source'] = pdf_path

    # 创建嵌入并加入向量数据库
    vectorstore.add_documents(docs)
    # 不需要调用 vectorstore.persist()，数据会自动持久化
    print(f"Processed and indexed {pdf_path}")

if __name__ == "__main__":
    # 在启动时处理现有的 PDF 文件
    for filename in os.listdir(WATCH_FOLDER):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(WATCH_FOLDER, filename)
            process_pdf(pdf_path)

    # 设置文件系统监视器
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=False)
    observer.start()
    print(f"Monitoring folder: {WATCH_FOLDER}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()