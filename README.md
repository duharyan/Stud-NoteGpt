# Stud_NoteGpt
StudNotesGpt is a Streamlit-based multimodal application that allows students to upload their notes in PPTX or PDF format and interact with them through a Retrieval-Augmented Generation (RAG) approach. Users can ask questions about their uploaded notes and get AI-powered answers.

## Features
- Upload PPTX or PDF files containing lecture notes.
- Extract text and images from uploaded files.
- Perform Optical Character Recognition (OCR) on extracted images.
- Store extracted content in a FAISS vector database.
- Query notes using AI models (Groq's LLaMA 3 or Google's Gemini 1.5 Flash).
- Get AI-generated responses based on document content.

## Installation
### Prerequisites
Ensure you have Python 3.8 or later installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root directory and add your API keys:

```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Usage
Run the application using the following command:

```sh
streamlit run app.py
```
![Screenshot 2025-01-30 034950](https://github.com/user-attachments/assets/f5b48c90-c1f1-409c-a829-fff38e7d9952)

![Screenshot 2025-01-30 030635](https://github.com/user-attachments/assets/bacc6dfa-94a1-4564-9ffa-8003f17b68ff)


## Contributing
Feel free to submit issues and pull requests to improve StudNotes.

