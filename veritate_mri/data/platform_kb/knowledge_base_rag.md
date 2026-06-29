# Grounding chat answers with your own documents

Veritate's chat page can ground its answers in documents you provide. This is retrieval-augmented generation, or RAG: before the model answers, the platform finds the most relevant passages from your documents and gives them to the model as context. This lets the model answer questions about your own material instead of relying only on what it learned during training.

The knowledge base is a folder of plain text files on your machine. You add documents by using the file uploader in the Knowledge section of the chat page's settings. Upload is instant: there is no embedding step and no model has to process the file first. Once a file is uploaded, the grounding toggle becomes available and you can turn it on.

Retrieval uses BM25, a classic lexical search method. It scores passages by how many of the same words they share with your question, then returns the top matches. Because it is lexical, it matches on shared keywords rather than on paraphrase, so the quality of grounding comes from the wording of the documents you upload. When grounding is on and the search finds matches, those passages are added to the prompt and the answer lists the sources it drew from.

The knowledge base is fully local and private. There is no embedding model, no external embedding service, and no Ollama dependency for chat grounding. The search runs entirely on your machine, in plain Python, over the files you uploaded. Your documents never leave your computer for the purpose of building or searching the index.

Grounding is opt-in per message. If you leave the toggle off, or if the knowledge base is empty, the chat just answers from the model alone with no retrieval. If you edit the knowledge-base files directly on disk rather than through the uploader, restart the dashboard (or upload again) so the search index picks up the change.
