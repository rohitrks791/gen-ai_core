Step: Document Loading

The system loads documents using document loaders
Goal: Convert raw files into document objects that can be processed.
you may clean the document as well like extra spaces, punctuation if necessary

step: Text Splitting (chunking)
Documents are usually too large for LLM context windows, so we split them into smaller chunks.
Chunking improves retrieval accuracy.

Step: Embedding Generation
Each chunk is converted into vector embedding.
Embedding models transform text into numerical vectors.

"Gradient Descent Optimization" => [0.23, -0.98, 0.552, ...]
These vectors represent semantic meaning(Every vector having separate meaning of the sentence that is captured)

step: vector database storage:
All embeddings are stored inside a vector database.
The vector db stores:
embeddings, original text chunks, metadata

RAG(Retrieval Augmented Generation)
Document Loader -> Text splitter -> Database -> Retrievers
1.User Asks a question , now the student interacts with the system.
2.Query Embedding: The question is also converted into an embedding.
3. Similarity Search: The vector database performs semantic similarity search.
Goal: Find chunks that are most relevant to the question.We search for similar embeddings in our vecotr db.
4. Retriever Component: The retriever selects the top-K relevant chunks.
These chunks form the context.
5. These context now goes to LLM . 
Based on the contenxt the LLM answrs and provide the result