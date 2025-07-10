# Legal RAG Backend

A FastAPI-based Retrieval-Augmented Generation (RAG) backend service for legal document queries using Google Gemini AI with clean citation extraction and structured responses.

 # Features

- **Document Processing**: Supports PDF, DOCX, and TXT legal documents
- **Smart Vector Search**: Uses cosine similarity for efficient document retrieval
- **AI-Powered Answers**: Generates responses using Google Gemini 2.5 Flash
- **Citation System**: Provides source attributions for transparency
- **Clean Web Interface**: Bootstrap-based UI for easy testing
- **RESTful API**: Well-documented endpoints with proper error handling
- **Real-time Processing**: Automatic document ingestion and embedding generation

## Tech Stack

- **Backend**: FastAPI (Python 3.11)
- **Embeddings**: Custom TF-IDF based embeddings (384-dimensional)
- **Vector Database**: Simple cosine similarity search
- **AI Model**: Google Gemini 2.5 Flash
- **Document Processing**: PyPDF2, python-docx, text file support
- **Frontend**: HTML, Bootstrap, Vanilla JavaScript

## API Endpoints

### POST /query
Query legal documents using natural language.

**Request:**
```json
{
  "query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
}
```

**Response:**
```json
{
  "answer": "No, an insurance company is not liable to pay compensation if a transport vehicle is used without a valid permit...",
  "citations": [
    {
      "text": "Use of a vehicle in a public place without a permit is a fundamental statutory infraction...",
      "source": "insurance_liability_case.txt"
    }
  ]
}
```

### GET /health
Check service health and initialization status.

### GET /documents/count
Get the number of processed documents.

##  Setup Instructions

### 1. Environment Setup

The application requires a Gemini API key. Get yours from [Google AI Studio](https://aistudio.google.com/).

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lexi.sg-rag-backend-test.git
cd lexi.sg-rag-backend-test

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Legal Documents

Place your legal documents in the `documents/` directory:
- Supported formats: PDF, DOCX, TXT
- Files will be automatically processed on startup

### 4. Run the Application

```bash
# Using Python directly
python run_server.py

# Or using uvicorn
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

### 5. Access the Application

- **Web Interface**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **Health Check**: http://localhost:5000/health

## Example Usage

### Using the Web Interface

1. Navigate to http://localhost:5000
2. Enter your legal query in the text area
3. Click "Submit Query"
4. Review the generated answer and citations

### Using the API

```bash
curl -X POST "http://localhost:5000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the insurance liability rules for vehicles without permits?"}'
```

## Testing

Run the test suite to validate functionality:

```bash
# Test core functionality
python test_app.py

# Test API endpoints
python test_api.py
```

##  Project Structure

```
legal-rag-backend/
├── app.py                 # Main FastAPI application
├── main.py               # Application entry point
├── documents/            # Legal document storage
├── services/             # Core business logic
│   ├── document_processor.py
│   ├── simple_embeddings.py
│   ├── simple_vector_store.py
│   └── gemini_service.py
├── models/               # Data models
│   └── schemas.py
├── static/              # Web interface assets
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/                # Processed document cache
└── tests/               # Test files
```

##  Features Implemented

 **Document Processing**: PDF, DOCX, TXT support with intelligent chunking  
 **Vector Search**: Cosine similarity-based document retrieval  
 **AI Integration**: Google Gemini 2.5 Flash for answer generation  
 **Citation System**: Automatic source attribution  
 **RESTful API**: Complete API with validation and error handling  
 **Web Interface**: Clean, responsive Bootstrap UI  
 **Caching**: Persistent document and embedding storage  
 **Health Monitoring**: Service status and document count tracking  

##  Sample Legal Documents

The application includes sample legal documents covering:
- Insurance liability cases
- Vehicle permit regulations
- Motor Vehicle Act provisions

##  Error Handling

The application includes comprehensive error handling for:
- Invalid API keys
- Document processing failures
- Vector search errors
- AI service timeouts
- Malformed requests

## Performance

- **Document Processing**: ~1000 words per chunk with 200-word overlap
- **Embedding Generation**: Custom TF-IDF with 384 dimensions
- **Search**: Top-5 similar documents retrieved
- **Response Time**: Typically 2-5 seconds for complex queries

##  Future Enhancements

- Support for additional document formats
- Advanced sentence-transformers embeddings
- FAISS integration for larger document collections
- Database persistence for production deployment
- User authentication and rate limiting

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.
