PDF Chatbot with Automated Embedding and Retrieval

This project provides a chatbot interface that allows users to ask questions based on the content of PDF files. The system automatically processes PDF files placed in a specific folder, generates embeddings using specified models, stores them in a vector database, and provides answers with source citations.

Features

	•	Automated PDF Processing: Automatically monitors a folder for new PDF files and processes them.
	•	Embedding Generation: Generates embeddings using specified models (e.g., cohere.embed-multilingual-v3).
	•	Vector Database Storage: Stores embeddings in a vector database (using Chroma).
	•	Reranking: Implements a reranker model (e.g., bge-reranker-base) to improve retrieval quality.
	•	Chat Interface: Provides a web-based chat interface for users to interact with the system.
	•	OpenAI-Compatible API: Offers an API endpoint compatible with OpenAI’s API format.
	•	Source Citation: Includes source documents in the chatbot’s responses.
	•	Configurable Chunk Size: Allows adjustment of chunk size and overlap during text splitting.

Prerequisites

	•	Python 3.9 or higher
	•	Required Python packages (see requirements.txt)

Installation

1. Clone the Repository

git clone https://github.com/tbdavid2019/rag-chat.git
cd rag-chat

2. Install Dependencies

Install the required Python packages:

pip install -r requirements.txt

3. Set Up Environment Variables

Create a .env file in the project root directory and add the following:

OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=http://openai.com/api/v1
CHAT_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
EMBEDDING_MODEL=cohere.embed-multilingual-v3
RERANKER_MODEL=bge-reranker-base
PDF_FOLDER=/content/data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

	•	Replace your-openai-api-key with your actual OpenAI API key.
	•	Adjust other settings as needed.

4. Prepare the PDF Folder

Ensure the folder specified in PDF_FOLDER exists and contains the PDF files you want to process.

Running the Application

1. Start the PDF Watcher

In one terminal window, run:

python pdf_watcher.py

This script monitors the specified folder for new PDF files and processes them automatically.

2. Start the Flask App

In another terminal window, run:

python app.py

This starts the Flask web application on http://localhost:7860.

Using the Application

Web Interface

Open a web browser and navigate to http://localhost:7860 to access the chat interface.
	•	Enter your query in the input box.
	•	Submit the query to receive an answer based on the content of your PDFs.
	•	The response includes citations of the source documents.

API Interface

You can interact with the chatbot via an API endpoint compatible with OpenAI’s API format.
	•	Endpoint: http://localhost:7860/v1/chat/completions
	•	Method: POST
	•	Request Format:

{
  "messages": [
    {"role": "user", "content": "Your question here"}
  ]
}


	•	Response Format:

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

project/
├── app.py                 # Main application
├── pdf_watcher.py         # Monitors folder and processes PDFs
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── templates/
│   └── chat.html          # Chat interface template

Configuration

Environment Variables

	•	OPENAI_API_KEY: Your OpenAI API key.
	•	OPENAI_API_BASE: Base URL for the OpenAI-compatible API gateway.
	•	CHAT_MODEL: The chat model to use (default: anthropic.claude-3-5-sonnet-20241022-v2:0).
	•	EMBEDDING_MODEL: The embedding model to use (default: cohere.embed-multilingual-v3).
	•	RERANKER_MODEL: The reranker model to use (default: bge-reranker-base).
	•	PDF_FOLDER: The folder to monitor for PDF files (default: /content/data).
	•	CHUNK_SIZE: The chunk size for text splitting (default: 1000).
	•	CHUNK_OVERLAP: The chunk overlap size (default: 200).

Adjusting Chunk Size

You can adjust the CHUNK_SIZE and CHUNK_OVERLAP in the .env file to control how the text from PDFs is split into chunks.

Dependencies

The project requires the following Python packages:
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

pip install -r requirements.txt

Contributing

Contributions are welcome! Please open an issue or submit a pull request.

License

This project is licensed under the MIT License.