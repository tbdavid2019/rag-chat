確定，以下是更新的 README.md，使用了您的 GitHub 儲存庫 tbdavid2019/rag-chat，並確保中文版本使用繁體中文。

RAG-CHAT

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded PDF documents. This project utilizes OpenAI’s GPT models, Hugging Face models for embeddings and re-ranking, and provides a Gradio interface for user interaction.

Features

	•	Document Upload and Processing: Automatically processes PDF documents and extracts text content.
	•	Retrieval-Augmented Generation: Uses vector embeddings and a retriever to find relevant documents for a given query.
	•	Re-ranking: Implements a re-ranking mechanism to improve the relevance of retrieved documents.
	•	Gradio Interface: Provides a user-friendly interface for interacting with the chatbot.
	•	Dockerized Deployment: Includes Docker and Docker Compose configurations for easy deployment.

Requirements

	•	Docker
	•	Docker Compose
	•	Access to OpenAI API
	•	Access to Hugging Face models (with appropriate access token if required)

Installation

	1.	Clone the Repository

```
git clone https://github.com/tbdavid2019/rag-chat.git
cd rag-chat
```

	2.	Set Up Environment Variables
Create a .env file in the root directory and add the following variables:
```
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
PDF_FOLDER=/content/data
CHROMA_DB_DIR=/chroma_db
HF_TOKEN=your-huggingface-access-token
```
	•	Replace your-openai-api-key with your actual OpenAI API key.
	•	Replace your-huggingface-access-token with your Hugging Face access token.
Note: Ensure that your .env file is not committed to version control for security reasons. Add .env to your .gitignore file.

	3.	Prepare the Data Directories
Create the necessary directories for data storage:
```
mkdir -p content/data
mkdir chroma_db
```
	•	Place your PDF files into the content/data directory.

	4.	Build and Run the Docker Containers
Build the Docker image and start the containers using Docker Compose:
```
docker-compose build
docker-compose up -d
```
The application should now be running and accessible.

Configuration

	•	Dockerfile: Contains instructions to build the Docker image.
	•	docker-compose.yml: Defines services, volumes, and environment variables for Docker Compose.
	•	supervisord.conf: Configures supervisord to run multiple processes within the Docker container.
	•	app.py: The main application script that starts the Gradio interface and processes PDFs.
	•	pdf_watcher.py: Script that watches for new PDFs and processes them.

Usage

	1.	Access the Gradio Interface
Open your browser and navigate to:
```
http://localhost:7860
```
Replace localhost with your server’s IP address if running remotely.

	2.	Interact with the Chatbot
	•	Enter your query in the text box.
	•	The chatbot will respond based on the content of the uploaded PDFs.

	3.	Add More PDFs
	•	Place additional PDF files into the content/data directory.
	•	The pdf_watcher.py script will automatically detect and process new PDFs.

License

This project is licensed under the MIT License.

Acknowledgments

	•	OpenAI for providing the GPT models.
	•	Hugging Face for providing transformer models.
	•	Gradio for the user interface framework.
	•	LangChain for building the retrieval and QA chains.

RAG-CHAT

一個基於檢索增強生成（RAG）的聊天機器人，能夠根據上傳的 PDF 文件回答問題。此項目使用了 OpenAI 的 GPT 模型、Hugging Face 的嵌入和重排序模型，並提供了 Gradio 介面供使用者互動。

特性

	•	文件上傳和處理：自動處理 PDF 文件並提取文本內容。
	•	檢索增強生成：使用向量嵌入和檢索器，根據查詢找到相關文件。
	•	重排序：實現了重排序機制，提高檢索文件的相關性。
	•	Gradio 介面：提供了使用者友好的介面，用於與聊天機器人互動。
	•	Docker 化部署：包含 Docker 和 Docker Compose 配置，方便部署。

要求

	•	Docker
	•	Docker Compose
	•	訪問 OpenAI API
	•	訪問 Hugging Face 模型（如果需要，需提供訪問權杖）

安裝

	1.	克隆儲存庫
```
git clone https://github.com/tbdavid2019/rag-chat.git
cd rag-chat
```

	2.	設定環境變數
在根目錄下創建 .env 文件，添加以下變數：
```
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
PDF_FOLDER=/content/data
CHROMA_DB_DIR=/chroma_db
HF_TOKEN=your-huggingface-access-token
```
	•	將 your-openai-api-key 替換為您的 OpenAI API 金鑰。
	•	將 your-huggingface-access-token 替換為您的 Hugging Face 訪問權杖。
注意：為了安全起見，請確保您的 .env 文件未被提交到版本控制系統。在 .gitignore 文件中添加 .env。

	3.	準備資料目錄
創建必要的目錄用於資料存儲：
```
mkdir -p content/data
mkdir chroma_db
```
	•	將您的 PDF 文件放入 content/data 目錄。

	4.	構建並運行 Docker 容器
使用 Docker Compose 構建映像並啟動容器：
```
docker-compose build
docker-compose up -d
```
應用程式現在應該已經啟動並可訪問。

配置

	•	Dockerfile：包含構建 Docker 映像的指令。
	•	docker-compose.yml：定義了 Docker Compose 的服務、卷和環境變數。
	•	supervisord.conf：配置了 supervisord，以便在 Docker 容器中運行多個進程。
	•	app.py：主應用程式腳本，啟動 Gradio 介面並處理 PDF。
	•	pdf_watcher.py：監視並處理新 PDF 的腳本。

使用方法

	1.	訪問 Gradio 介面
在瀏覽器中打開：

http://localhost:7860

如果在遠端運行，請將 localhost 替換為您的服務器 IP 地址。

	2.	與聊天機器人互動
	•	在文本框中輸入您的查詢。
	•	聊天機器人將根據上傳的 PDF 內容進行回答。
	3.	添加更多 PDF
	•	將新的 PDF 文件放入 content/data 目錄。
	•	pdf_watcher.py 腳本會自動檢測並處理新的 PDF。

授權許可

本項目使用 MIT 許可證。

致謝

	•	感謝 OpenAI 提供的 GPT 模型。
	•	感謝 Hugging Face 提供的 Transformer 模型。
	•	感謝 Gradio 提供的使用者介面框架。
	•	感謝 LangChain 提供的檢索和問答鏈構建工具。

	