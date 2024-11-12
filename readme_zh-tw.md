PDF Chatbot with Automated Embedding and Retrieval

基於 PDF 的自動嵌入和檢索聊天機器人

This project provides a chatbot interface that allows users to ask questions based on the content of PDF files. The system automatically processes PDF files placed in a specific folder, generates embeddings using specified models, stores them in a vector database, and provides answers with source citations.

本專案提供一個聊天機器人介面，允許使用者根據 PDF 文件的內容提出問題。系統會自動處理放置在特定資料夾中的 PDF 檔案，使用指定的模型生成嵌入，將其存儲在向量資料庫中，並提供帶有來源引用的答案。

Features

功能特點

	•	Automated PDF Processing: Automatically monitors a folder for new PDF files and processes them.
	•	自動化 PDF 處理：自動監控資料夾中的新 PDF 檔案並進行處理。
	•	Embedding Generation: Generates embeddings using specified models (e.g., cohere.embed-multilingual-v3).
	•	生成嵌入：使用指定的模型生成嵌入（例如 cohere.embed-multilingual-v3）。
	•	Vector Database Storage: Stores embeddings in a vector database (using Chroma).
	•	向量資料庫存儲：將嵌入存儲在向量資料庫中（使用 Chroma）。
	•	Reranking: Implements a reranker model (e.g., bge-reranker-base) to improve retrieval quality.
	•	重排序：實施重排序模型（例如 bge-reranker-base）以提升檢索品質。
	•	Chat Interface: Provides a web-based chat interface for users to interact with the system.
	•	聊天介面：提供基於網頁的聊天介面供使用者與系統互動。
	•	OpenAI-Compatible API: Offers an API endpoint compatible with OpenAI’s API format.
	•	OpenAI 相容的 API：提供與 OpenAI API 格式相容的 API 端點。
	•	Source Citation: Includes source documents in the chatbot’s responses.
	•	來源引用：在聊天機器人的回應中包含來源文件。
	•	Configurable Chunk Size: Allows adjustment of chunk size and overlap during text splitting.
	•	可配置的分塊大小：允許在文本分割時調整分塊大小和重疊。

Prerequisites

先決條件

	•	Python 3.9 or higher
	•	Python 3.9 或更高版本
	•	OpenAI API key
	•	OpenAI API 金鑰
	•	Required Python packages (see requirements.txt)
	•	必要的 Python 套件（見 requirements.txt）

Installation

安裝

1. Clone the Repository

1. 複製儲存庫

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2. Install Dependencies

2. 安裝依賴

Install the required Python packages:
安裝必要的 Python 套件：

pip install -r requirements.txt

3. Set Up Environment Variables

3. 設定環境變數

Create a .env file in the project root directory and add the following:
在專案根目錄下建立一個 .env 檔案，並添加以下內容：

OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=http://Bedroc-Proxy-RCrbdJs2OHgQ-1123811820.us-west-2.elb.amazonaws.com/api/v1
CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
EMBEDDING_MODEL=cohere.embed-multilingual-v3
RERANKER_MODEL=bge-reranker-base
PDF_FOLDER=/content/data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

	•	Replace your-openai-api-key with your actual OpenAI API key.
	•	將 your-openai-api-key 替換為您的實際 OpenAI API 金鑰。
	•	Adjust other settings as needed.
	•	根據需要調整其他設定。

4. Prepare the PDF Folder

4. 準備 PDF 資料夾

Ensure the folder specified in PDF_FOLDER exists and contains the PDF files you want to process.
確保在 PDF_FOLDER 中指定的資料夾存在，並包含您要處理的 PDF 檔案。

Running the Application

運行應用程式

1. Start the PDF Watcher

1. 啟動 PDF 監控器

In one terminal window, run:
在一個終端窗口中，運行：

python pdf_watcher.py

This script monitors the specified folder for new PDF files and processes them automatically.
此腳本會監控指定的資料夾中的新 PDF 檔案，並自動處理它們。

2. Start the Flask App

2. 啟動 Flask 應用

In another terminal window, run:
在另一個終端窗口中，運行：

python app.py

This starts the Flask web application on http://localhost:5000.
這將在 http://localhost:5000 啟動 Flask 網頁應用。

Using the Application

使用應用程式

Web Interface

網頁介面

Open a web browser and navigate to http://localhost:5000 to access the chat interface.
打開網頁瀏覽器並訪問 http://localhost:5000 以訪問聊天介面。
	•	Enter your query in the input box.
	•	在輸入框中輸入您的問題。
	•	Submit the query to receive an answer based on the content of your PDFs.
	•	提交問題，將根據您的 PDF 內容獲得答案。
	•	The response includes citations of the source documents.
	•	回應中將包含來源文件的引用。

API Interface

API 介面

You can interact with the chatbot via an API endpoint compatible with OpenAI’s API format.
您可以通過與 OpenAI API 格式相容的 API 端點與聊天機器人互動。
	•	Endpoint: http://localhost:5000/v1/chat/completions
	•	端點：http://localhost:5000/v1/chat/completions
	•	Method: POST
	•	方法：POST
	•	Request Format:
	•	請求格式：

{
  "messages": [
    {"role": "user", "content": "Your question here"}
  ]
}


	•	Response Format:
	•	回應格式：

{
  "id": "unique-id",
  "object": "chat.completion",
  "created": timestamp,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The assistant's response.\n\nSource: source1.pdf, source2.pdf"
      },
      "finish_reason": "stop"
    }
  ]
}



Project Structure

專案結構

project/
├── app.py                 # Main application
├── pdf_watcher.py         # Monitors folder and processes PDFs
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── templates/
│   └── chat.html          # Chat interface template

Configuration

配置

Environment Variables

環境變數

	•	OPENAI_API_KEY: Your OpenAI API key.
	•	OPENAI_API_KEY：您的 OpenAI API 金鑰。
	•	OPENAI_API_BASE: Base URL for the OpenAI-compatible API gateway.
	•	OPENAI_API_BASE：OpenAI 相容 API 閘道的基礎 URL。
	•	CHAT_MODEL: The chat model to use (default: anthropic.claude-3-5-sonnet-20241022-v2:0).
	•	CHAT_MODEL：要使用的聊天模型（預設值：anthropic.claude-3-5-sonnet-20241022-v2:0）。
	•	EMBEDDING_MODEL: The embedding model to use (default: cohere.embed-multilingual-v3).
	•	EMBEDDING_MODEL：要使用的嵌入模型（預設值：cohere.embed-multilingual-v3）。
	•	RERANKER_MODEL: The reranker model to use (default: bge-reranker-base).
	•	RERANKER_MODEL：要使用的重排序模型（預設值：bge-reranker-base）。
	•	PDF_FOLDER: The folder to monitor for PDF files (default: /content/data).
	•	PDF_FOLDER：要監控的 PDF 檔案資料夾（預設值：/content/data）。
	•	CHUNK_SIZE: The chunk size for text splitting (default: 1000).
	•	CHUNK_SIZE：文本分割的分塊大小（預設值：1000）。
	•	CHUNK_OVERLAP: The chunk overlap size (default: 200).
	•	CHUNK_OVERLAP：分塊重疊大小（預設值：200）。

Adjusting Chunk Size

調整分塊大小

You can adjust the CHUNK_SIZE and CHUNK_OVERLAP in the .env file to control how the text from PDFs is split into chunks.
您可以在 .env 檔案中調整 CHUNK_SIZE 和 CHUNK_OVERLAP，以控制 PDF 文本如何被分割成塊。

Dependencies

依賴

The project requires the following Python packages:
本專案需要以下 Python 套件：
	•	langchain
	•	openai
	•	faiss-cpu
	•	flask
	•	transformers
	•	sentence_transformers
	•	chromadb
	•	python-dotenv
	•	watchdog

Install them using:
使用以下命令安裝：

pip install -r requirements.txt

Contributing

貢獻

Contributions are welcome! Please open an issue or submit a pull request.
歡迎貢獻！請提交問題或拉取請求。

License

授權

This project is licensed under the MIT License.
本專案採用 MIT 授權。