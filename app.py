import os
import requests
import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import gradio as gr
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

# 加载环境变量
load_dotenv()



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



# 获取环境变量或使用默认值
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("未找到 OPENAI_API_KEY，请在 .env 文件中设置。")

OPENAI_API_BASE = os.getenv(
    "OPENAI_API_BASE",
    "http://your-api-endpoint/api/v1"
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "your-model-name")  # 替换为您的模型名称
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "your-embedding-model")  # 替换为您的嵌入模型
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
PDF_FOLDER = os.getenv('PDF_FOLDER', '/content/data')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', './chroma_db')
# 获取 Hugging Face 访问令牌
HF_TOKEN = os.getenv('HF_TOKEN')

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
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()

# 初始化重排序模型
tokenizer = AutoTokenizer.from_pretrained(
    RERANKER_MODEL,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)
reranker_model = AutoModel.from_pretrained(
    RERANKER_MODEL,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reranker_model.to(device)

def rerank_documents(query, docs):
    if not docs:
        return []

    # 获取查询的嵌入
    query_inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_embedding = reranker_model(**query_inputs).last_hidden_state.mean(dim=1)

    # 获取文档的嵌入
    doc_texts = [doc.page_content for doc in docs]
    if not doc_texts:
        return []

    doc_inputs = tokenizer(doc_texts, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        doc_embeddings = reranker_model(**doc_inputs).last_hidden_state.mean(dim=1)

    # 计算余弦相似度
    scores = F.cosine_similarity(query_embedding, doc_embeddings)

    # 根据分数排序文档
    ranked_docs = [doc for _, doc in sorted(zip(scores.cpu(), docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs

def generate_answer(question):
    # 检索相关文档
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "抱歉，我未能找到相關的文檔來回答您的問題。"

    # 重排序
    ranked_docs = rerank_documents(question, docs)

    # 构建上下文
    context = "\n".join([doc.page_content for doc in ranked_docs])

    # 构建请求数据
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一個樂於助人的助理，請用繁體中文回答問題。"
            },
            {
                "role": "user",
                "content": f"基於以下內容回答問題：\n{context}\n\n問題：{question}"
            }
        ],
        "temperature": 0.7,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # 发送请求到 API
    response = requests.post(
        f"{OPENAI_API_BASE}/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        sources = [doc.metadata['source'] for doc in ranked_docs]
        return answer + "\n\n來源：" + ", ".join(sources)
    else:
        return f"请求失败，状态码：{response.status_code}，信息：{response.text}"

# 构建 Gradio 界面
iface = gr.Interface(
    fn=generate_answer,
    inputs="text",
    outputs="text",
    title="文檔問答系統",
    description="基於上傳的 PDF 文檔進行問答。"
)

# 处理 PDF 文档
def process_pdfs():
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, filename)
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
            print(f"Processed and indexed {pdf_path}")

if __name__ == '__main__':
    # 首先处理 PDF 文档
    process_pdfs()
    # 启动 Gradio 接口，设置 share=True 以生成公共链接
    iface.launch(server_name="0.0.0.0", server_port=7860)
    