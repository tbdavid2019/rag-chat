rag-chat

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

git clone https://github.com/tbdavid2019/rag-chat.git
cd rag-chat


	2.	Set Up Environment Variables
Create a .env file in the root directory and add the following variables:

OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
PDF_FOLDER=/content/data
CHROMA_DB_DIR=/chroma_db
HF_TOKEN=your-huggingface-access-token

	•	Replace your-openai-api-key with your actual OpenAI API key.
	•	Replace your-huggingface-access-token with your Hugging Face access token.
Note: Ensure that your .env file is not committed to version control for security reasons. Add .env to your .gitignore file.

	3.	Prepare the Data Directories
Create the necessary directories for data storage:

mkdir -p content/data
mkdir chroma_db

	•	Place your PDF files into the content/data directory.

	4.	Build and Run the Docker Containers
Build the Docker image and start the containers using Docker Compose:

docker-compose build
docker-compose up -d

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

http://localhost:7860

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

rag-chat

一个基于检索增强生成（RAG）的聊天机器人，能够根据上传的 PDF 文档回答问题。该项目使用了 OpenAI 的 GPT 模型、Hugging Face 的嵌入和重排序模型，并提供了 Gradio 界面供用户交互。

特性

	•	文档上传和处理：自动处理 PDF 文档并提取文本内容。
	•	检索增强生成：使用向量嵌入和检索器，根据查询找到相关文档。
	•	重排序：实现了重排序机制，提升检索文档的相关性。
	•	Gradio 界面：提供了用户友好的界面用于与聊天机器人交互。
	•	Docker 化部署：包含 Docker 和 Docker Compose 配置，便于部署。

要求

	•	Docker
	•	Docker Compose
	•	访问 OpenAI API
	•	访问 Hugging Face 模型（如果需要，需提供访问令牌）

安装

	1.	克隆仓库

git clone https://github.com/tbdavid2019/rag-chat.git
cd rag-chat


	2.	设置环境变量
在根目录下创建 .env 文件，添加以下变量：

OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
PDF_FOLDER=/content/data
CHROMA_DB_DIR=/chroma_db
HF_TOKEN=your-huggingface-access-token

	•	将 your-openai-api-key 替换为您的 OpenAI API 密钥。
	•	将 your-huggingface-access-token 替换为您的 Hugging Face 访问令牌。
注意：为了安全，请确保您的 .env 文件未被提交到版本控制系统。将 .env 添加到 .gitignore 文件中。

	3.	准备数据目录
创建用于存储数据的必要目录：

mkdir -p content/data
mkdir chroma_db

	•	将您的 PDF 文件放置在 content/data 目录中。

	4.	构建并运行 Docker 容器
使用 Docker Compose 构建镜像并启动容器：

docker-compose build
docker-compose up -d

应用程序现在应该已经启动并可以访问。

配置

	•	Dockerfile：包含构建 Docker 镜像的指令。
	•	docker-compose.yml：定义了 Docker Compose 的服务、卷和环境变量。
	•	supervisord.conf：配置了 supervisord，以便在 Docker 容器中运行多个进程。
	•	app.py：主应用程序脚本，启动 Gradio 界面并处理 PDF。
	•	pdf_watcher.py：监视新 PDF 并处理的脚本。

使用方法

	1.	访问 Gradio 界面
在浏览器中打开：

http://localhost:7860

如果在远程运行，请将 localhost 替换为您的服务器 IP 地址。

	2.	与聊天机器人交互
	•	在文本框中输入您的查询。
	•	聊天机器人将根据上传的 PDF 内容进行回答。
	3.	添加更多 PDF
	•	将额外的 PDF 文件放入 content/data 目录。
	•	pdf_watcher.py 脚本会自动检测并处理新的 PDF。

许可证

本项目使用 MIT 许可证。

致谢

	•	感谢 OpenAI 提供的 GPT 模型。
	•	感谢 Hugging Face 提供的 Transformer 模型。
	•	感谢 Gradio 提供的用户界面框架。
	•	感谢 LangChain 提供的检索和问答链构建工具。