# 📘 Chatbot RAG using Streamlit

A Retrieval-Augmented Generation (**RAG**) chatbot built with **Streamlit**, **LangChain**, and **OpenAI**.
This app indexes multiple external course-related websites and allows users to ask questions, receiving accurate answers based on real website content.

🔗 **Live Demo** (Jupyter Notebook): `rag-demo.ipynb`
🔗 **Repository:** [https://github.com/1FarukDev/Chatbot-RAG-using-streamlit](https://github.com/1FarukDev/Chatbot-RAG-using-streamlit)

---

## 🚀 Features

* **Streamlit UI** for interactive chat experience
* **RAG Pipeline** using LangChain
* **Web scraping** using `UnstructuredURLLoader`
* **Text splitting** with `RecursiveCharacterTextSplitter`
* **Embedding storage** using **ChromaDB**
* **OpenAI LLM** for generating accurate answers
* **Safety prompt** to prevent hallucination
* Answers only using the provided context
* Clean and simple architecture in `app.py`

---

## 🧠 Project Architecture

```
Web URLs → Loader → Splitting → Embeddings → Chroma Vectorstore
                ↑                                 ↓
             Query ← Retriever ← LLM (OpenAI) ← Prompt Template
```

### Detailed Flow

1. **Load data** from multiple course-related URLs
2. **Split documents** into small chunks
3. **Generate embeddings** using `OpenAIEmbeddings()`
4. **Store and search** chunks in **ChromaDB**
5. **Retrieve context** related to user query
6. **Feed context + query** to the LLM
7. **Return accurate, context-aware response**

---

## 📂 Project Structure

```
.
├── app.py                # Streamlit application
├── rag-demo.ipynb        # Notebook demo
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **LangChain**
* **OpenAI API**
* **ChromaDB**
* **Unstructured.io Loader**
* **dotenv**

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/1FarukDev/Chatbot-RAG-using-streamlit
cd Chatbot-RAG-using-streamlit
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Your app will open at:

```
http://localhost:8501
```

---

## 🧩 Code Overview (`app.py`)

The core logic does the following:

* Load multiple URLs
* Process text with LangChain
* Embed into Chroma
* Retrieve relevant context
* Generate an answer with OpenAI
* Display it in Streamlit chat UI

Your final block:

```python
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    st.write(response['answer'])
```

---

## 📘 Requirements

```
langchain
langchain_community
langchain_openai
python-dotenv
streamlit
langchain_experimental
sentence-transformers
langchain_chroma
langchainhub
unstructured
```

---

## 🎯 Future Improvements

* Add **persistent** Chroma storage
* Add **UI styling** for chat interface
* Integrate **streaming LLM responses**
* Allow users to upload their documents
* Add logging + monitoring

---

## 🤝 Contributing

Feel free to open issues, suggest improvements, or submit PRs.

---

## 📜 License

MIT License.
