# Interview Guide: PDF Knowledge Assistant

This guide is designed to help you confidently explain every aspect of this project during a software engineering interview (e.g., for Infosys, TCS, or general Full Stack / AI roles).

## 1. Core Concepts

### What is RAG?
**Retrieval-Augmented Generation (RAG)** is a technique that gives large language models (LLMs) the ability to access specific, private, or up-to-date information that they weren't originally trained on. Instead of relying on the LLM's internal memory to answer a question, we *retrieve* relevant documents from our own database and *augment* the prompt with those documents before *generating* the answer.

### What is LangChain?
LangChain is a framework that makes it easier to build applications powered by LLMs. It provides standardized tools for loading documents, chunking text, connecting to vector databases, and constructing prompts. In this project, we used LangChain to orchestrate the pipeline cleanly.

### What is FAISS?
**Facebook AI Similarity Search (FAISS)** is an open-source library developed by Meta. It allows developers to quickly search for embeddings (vectors) that are similar to each other. We use it as our in-memory vector database to find the text chunks most relevant to the user's question.

### What are Embeddings?
Embeddings are numerical representations of text. An embedding model reads a chunk of text and outputs a long list of numbers (a vector). The mathematical distance between two vectors represents how similar the texts are in meaning. For example, "dog" and "puppy" will have vectors that are very close to each other.

### Why do we use Chunking?
LLMs have a "context window" limit—they can only read a certain number of words at a time. Furthermore, sending a 100-page document to an LLM is expensive and can confuse the model. By breaking the text into smaller chunks (e.g., 700 characters), we can search our database for only the 3 or 4 paragraphs that actually contain the answer, keeping the prompt small and highly relevant.

### Why use Overlap in Chunking?
If we split text without overlap, a sentence might get cut in half, destroying its meaning. By using an overlap (e.g., 120 characters), the end of Chunk A is repeated at the beginning of Chunk B, ensuring that context isn't lost across chunk boundaries.

### Why explicitly code the RAG pipeline instead of using `create_retrieval_chain`?
In LangChain, high-level chains hide the internal logic. By writing the retrieval, context formatting, and LLM invocation explicitly in `src/rag.py`, I have full control over the prompt format, error handling, and data flow. It demonstrates that I understand *how* RAG works, rather than just knowing how to call a wrapper function.

---

## 2. Common Interview Questions & Answers

### Q: Walk me through the architecture of your project. What happens from upload to answer?
**A:** 
1. **Upload:** The user uploads a PDF via the Streamlit UI. The file is temporarily saved to the server.
2. **Extraction:** `PyPDFLoader` reads the file and extracts the text page by page.
3. **Chunking:** `RecursiveCharacterTextSplitter` breaks the text into 700-character chunks.
4. **Embedding & Storage:** We send these chunks to OpenAI's embedding model (`text-embedding-3-small`) to convert them into vectors, and store them in a FAISS vector database.
5. **Querying:** When the user asks a question, we embed the question using the same OpenAI model.
6. **Retrieval:** We perform a similarity search in FAISS to find the top 4 most relevant chunks.
7. **Generation:** We insert those chunks into a strict Prompt Template along with the user's question, and send it to OpenAI's GPT (e.g., `gpt-4.1-mini` or `gpt-4o-mini`).
8. **Display:** We display the generated answer and the source chunks in the Streamlit UI.

### Q: Why did you use Python logging instead of print statements?
**A:** `print()` is acceptable for quick scripts, but in production, `logging` is essential. It allows us to categorize messages by severity (INFO, WARNING, ERROR), formats them with timestamps and module names, and allows us to easily redirect logs to monitoring tools. It makes debugging much easier if the app fails in production (e.g., on Streamlit Cloud).

### Q: How do you prevent the LLM from hallucinating?
**A:** Two ways. First, through the architecture (RAG), which provides factual context. Second, through a strict System Prompt. I explicitly wrote in `src/prompts.py`: *"If the answer is not contained within the provided context, clearly state: 'I don't know'."* I also set the LLM's `temperature` parameter to 0.0, making its responses highly deterministic and less creative.

### Q: How did you handle environment variables and security?
**A:** I used the `python-dotenv` library. The OpenAI API key is stored in a `.env` file, which is explicitly ignored by Git (via `.gitignore`) so it never gets pushed to GitHub. In production (Streamlit Cloud), the key is securely injected using Streamlit Secrets. Furthermore, my `src/config.py` explicitly checks for the key at runtime and logs a warning if it's missing.

### Q: Why Streamlit? Why not React/Node.js?
**A:** Streamlit allows Python developers to build interactive, data-driven web applications extremely fast without writing frontend code. For AI and data science prototypes, the time-to-market is unbeatable. If this were a consumer-facing app with thousands of concurrent users, I would separate the backend into FastAPI and build a React frontend, but for a portfolio RAG project, Streamlit is the industry standard.

---

## 3. Advanced / Deep Dive Questions

### Q: What is the time complexity of the similarity search in FAISS?
**A:** By default, FAISS uses an exact search (Flat L2 index), which has a time complexity of `O(N * D)`, where N is the number of chunks and D is the dimensionality of the embeddings. For small documents, this is instantaneous. If we scaled this to millions of documents, we would switch FAISS to use an IVF (Inverted File) index or HNSW (Hierarchical Navigable Small World) graph to reduce the search time to `O(log N)`.

### Q: What happens if two chunks have contradicting information?
**A:** The LLM will see both chunks in the context prompt. How it responds depends entirely on the system prompt. Since my prompt says "If the context is ambiguous, explain the ambiguity based only on the text," GPT should ideally respond by noting that the document contains conflicting statements.

### Q: Where are the PDFs actually stored when a user uploads them, and for how long?
**A:** When a user uploads a PDF, it is held in memory by Streamlit. In my code (`src/loader.py`), I explicitly use Python's `tempfile.NamedTemporaryFile` to write the PDF to the server's temporary `/tmp` directory just long enough for `PyPDFLoader` to read it. Once the extraction is done, the temporary file is immediately deleted from the disk. Therefore, the PDFs are not permanently stored anywhere.

### Q: How long are the embeddings/vectors stored? What is the time limit?
**A:** The vectors are stored strictly in Streamlit's **Session State** (`st.session_state`). Session State is tied exclusively to the user's current browser tab. The vectors exist in the server's RAM only for as long as that specific browser tab remains open and active. If the user refreshes the page, closes the tab, or if the app goes to sleep due to inactivity, the vectors are completely destroyed.

### Q: What happens if 10 concurrent users open the app simultaneously and upload different PDFs? Can one user see another user's data?
**A:** No, data bleeding is impossible in this architecture. This is because Streamlit handles every connected user in a completely isolated thread with its own unique `st.session_state`. 
If 10 users open the app, Streamlit spins up 10 isolated sessions. User A's FAISS vector store is stored in User A's session state, and User B's FAISS store is in User B's session state. They run concurrently in server RAM but never intersect, ensuring complete data privacy between concurrent users.

---

## 🚀 Final Advice for the Candidate

### 4. Project Limitations & Future Scope
If an interviewer asks "What would you improve?", this is how you answer:

1. **No Memory:** The current implementation is single-turn. If I ask "What is X?", and then follow up with "Tell me more about it", the system will fail because it doesn't remember the previous question. **Improvement:** Add LangChain's `ConversationBufferMemory` to pass chat history into the LLM prompt.
2. **Tables and Images:** `PyPDFLoader` is great for text but ruins table structures and ignores images. **Improvement:** Integrate unstructured.io or a vision model to handle complex PDFs.
3. **Chunking Strategy:** Fixed-size chunking might split a paragraph exactly in the middle of a crucial thought. **Improvement:** Implement "Semantic Chunking", which uses an LLM to find logical breakpoint sentences rather than relying on strict character counts.
