"""
RAG (Retrieval Augmented Generation) Engine for LocalWrite.

Enhanced version with:
- Better embedding model (bge-base-en-v1.5)
- Smart sentence-based chunking
- Hybrid search (semantic + BM25 keyword matching)
"""

import os
import re
import hashlib
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass

# Lazy imports for optional dependencies
chromadb = None
SentenceTransformer = None
BM25Okapi = None


def _load_dependencies():
    """Lazy load heavy dependencies."""
    global chromadb, SentenceTransformer, BM25Okapi

    if chromadb is None:
        try:
            import chromadb as _chromadb
            chromadb = _chromadb
        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

    if BM25Okapi is None:
        try:
            from rank_bm25 import BM25Okapi as _BM25
            BM25Okapi = _BM25
        except ImportError:
            # BM25 is optional - will fall back to semantic-only search
            BM25Okapi = None


@dataclass
class Document:
    """Represents a stored document."""
    id: str
    name: str
    path: str
    chunk_count: int
    word_count: int
    added_date: str


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its relevance score."""
    content: str
    document_name: str
    document_id: str
    score: float


class RAGEngine:
    """
    Enhanced RAG Engine for document-based question answering.

    Features:
    - BGE embedding model (state-of-the-art for retrieval)
    - Smart sentence-based chunking
    - Hybrid search (semantic + BM25)
    - Persistent vector storage
    """

    # Better embedding model - optimized for retrieval
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

    # Chunking parameters
    CHUNK_SIZE = 512  # target tokens per chunk
    CHUNK_OVERLAP = 2  # sentences overlap
    MIN_CHUNK_SIZE = 100  # minimum characters per chunk

    # Retrieval parameters
    TOP_K = 8  # number of chunks to retrieve
    SEMANTIC_WEIGHT = 0.7  # weight for semantic search
    BM25_WEIGHT = 0.3  # weight for keyword search

    def __init__(self, storage_path: str = None):
        """
        Initialize RAG Engine.

        Args:
            storage_path: Path to store vector database. Defaults to ~/.localwrite/rag_db
        """
        self.storage_path = storage_path or os.path.expanduser("~/.localwrite/rag_db")
        os.makedirs(self.storage_path, exist_ok=True)

        self._embedder = None
        self._client = None
        self._collection = None
        self._bm25_index = None
        self._chunk_texts = []  # For BM25 indexing
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of heavy components."""
        if self._initialized:
            return

        _load_dependencies()

        # Initialize embedding model (BGE requires specific prefix for queries)
        self._embedder = SentenceTransformer(self.EMBEDDING_MODEL)

        # Initialize ChromaDB with persistent storage
        self._client = chromadb.PersistentClient(path=self.storage_path)

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name="documents_v2",  # New collection for better embeddings
            metadata={"hnsw:space": "cosine"}
        )

        # Build BM25 index from existing documents
        self._rebuild_bm25_index()

        self._initialized = True

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s', r'\1<DOT> ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _chunk_text_smart(self, text: str) -> List[str]:
        """
        Smart chunking that respects sentence boundaries.

        Creates chunks of approximately CHUNK_SIZE tokens while:
        - Never breaking mid-sentence
        - Maintaining overlap for context
        - Respecting paragraph boundaries when possible
        """
        # First split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        all_sentences = []
        for para in paragraphs:
            sentences = self._split_into_sentences(para)
            all_sentences.extend(sentences)
            # Add paragraph marker
            if sentences:
                all_sentences.append("<PARA_BREAK>")

        # Remove trailing paragraph break
        if all_sentences and all_sentences[-1] == "<PARA_BREAK>":
            all_sentences.pop()

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in all_sentences:
            if sentence == "<PARA_BREAK>":
                # If current chunk is substantial, consider it a natural break point
                if current_length > self.MIN_CHUNK_SIZE:
                    chunks.append(' '.join(current_chunk))
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-self.CHUNK_OVERLAP:] if len(current_chunk) > self.CHUNK_OVERLAP else []
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)
                continue

            sentence_length = len(sentence)

            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > self.CHUNK_SIZE * 4:  # ~4 chars per token estimate
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-self.CHUNK_OVERLAP:] if len(current_chunk) > self.CHUNK_OVERLAP else []
                    current_chunk = overlap_sentences
                    current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.MIN_CHUNK_SIZE:
                chunks.append(chunk_text)
            elif chunks:
                # Append to previous chunk if too small
                chunks[-1] = chunks[-1] + ' ' + chunk_text

        return chunks if chunks else [text]  # Return original if no chunks created

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID from file path."""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from stored documents."""
        if BM25Okapi is None:
            return

        all_data = self._collection.get()
        if not all_data or not all_data['documents']:
            self._bm25_index = None
            self._chunk_texts = []
            return

        self._chunk_texts = all_data['documents']
        # Tokenize for BM25
        tokenized_chunks = [doc.lower().split() for doc in self._chunk_texts]
        self._bm25_index = BM25Okapi(tokenized_chunks)

    def _embed_text(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for texts.

        BGE models work better with instruction prefixes.
        """
        if is_query:
            # BGE recommends this prefix for queries
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        embeddings = self._embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    # Document size limits
    MAX_FILE_SIZE_MB = 100  # Hard limit
    WARN_FILE_SIZE_MB = 50  # Warning threshold
    EMBEDDING_BATCH_SIZE = 32  # Process embeddings in batches

    def add_document(
        self,
        file_path: str,
        content: str,
        progress_callback: Callable[[str], None] = None
    ) -> Document:
        """
        Add a document to the RAG database with production-grade processing.

        Features:
        - File size validation
        - Batch embedding generation (memory efficient)
        - Detailed progress tracking
        - Chunk validation
        """
        import gc

        self._ensure_initialized()

        doc_id = self._generate_doc_id(file_path)
        doc_name = os.path.basename(file_path)

        if progress_callback:
            progress_callback(f"Processing: {doc_name}")

        # === 1. Validate document size ===
        content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)

        if content_size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(f"Document too large ({content_size_mb:.1f}MB). Maximum size is {self.MAX_FILE_SIZE_MB}MB.")

        if content_size_mb > self.WARN_FILE_SIZE_MB:
            if progress_callback:
                progress_callback(f"⚠️ Large document ({content_size_mb:.1f}MB) - processing may take a while...")

        # === 2. Check if document already exists ===
        existing = self._collection.get(where={"document_id": doc_id})
        if existing and existing['ids']:
            self._collection.delete(where={"document_id": doc_id})
            if progress_callback:
                progress_callback(f"Updating existing document: {doc_name}")

        # === 3. Smart chunking ===
        if progress_callback:
            progress_callback("Analyzing document structure...")
        chunks = self._chunk_text_smart(content)

        # Validate chunks - remove empty or too small
        chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) >= 50]

        if not chunks:
            raise ValueError("Document is empty or contains no meaningful content")

        total_chunks = len(chunks)
        if progress_callback:
            progress_callback(f"Created {total_chunks} chunks")

        # === 4. Batch embedding generation (memory efficient) ===
        all_embeddings = []

        for batch_start in range(0, total_chunks, self.EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + self.EMBEDDING_BATCH_SIZE, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]

            if progress_callback:
                progress_callback(f"Embedding chunks {batch_start + 1}-{batch_end} of {total_chunks}...")

            batch_embeddings = self._embed_text(batch_chunks, is_query=False)
            all_embeddings.extend(batch_embeddings)

            # Memory cleanup after each batch
            gc.collect()

        # === 5. Prepare metadata ===
        ids = [f"{doc_id}_{i}" for i in range(total_chunks)]
        metadatas = [
            {
                "document_id": doc_id,
                "document_name": doc_name,
                "chunk_index": i,
                "chunk_size": len(chunks[i])
            }
            for i in range(total_chunks)
        ]

        # === 6. Store in vector database (batch insert for large docs) ===
        if progress_callback:
            progress_callback("Storing in vector database...")

        # Batch insert for very large documents
        if total_chunks > 500:
            batch_size = 100
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                self._collection.add(
                    ids=ids[i:end_idx],
                    embeddings=all_embeddings[i:end_idx],
                    documents=chunks[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                if progress_callback:
                    progress_callback(f"Stored {end_idx}/{total_chunks} chunks...")
        else:
            self._collection.add(
                ids=ids,
                embeddings=all_embeddings,
                documents=chunks,
                metadatas=metadatas
            )

        # === 7. Rebuild BM25 index ===
        if progress_callback:
            progress_callback("Building keyword index...")
        self._rebuild_bm25_index()

        # === 8. Final cleanup ===
        del all_embeddings
        gc.collect()

        # Create document record
        from datetime import datetime
        word_count = len(content.split())
        doc = Document(
            id=doc_id,
            name=doc_name,
            path=file_path,
            chunk_count=total_chunks,
            word_count=word_count,
            added_date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

        if progress_callback:
            progress_callback(f"✓ Added: {doc_name} ({total_chunks} chunks, {word_count:,} words)")

        return doc

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its embeddings from the database."""
        self._ensure_initialized()

        existing = self._collection.get(where={"document_id": doc_id})
        if not existing or not existing['ids']:
            return False

        self._collection.delete(where={"document_id": doc_id})

        # Rebuild BM25 index
        self._rebuild_bm25_index()

        return True

    def get_documents(self) -> List[Document]:
        """Get list of all stored documents."""
        self._ensure_initialized()

        all_data = self._collection.get()

        if not all_data or not all_data['metadatas']:
            return []

        # Group by document_id
        docs_dict = {}
        for metadata in all_data['metadatas']:
            doc_id = metadata['document_id']
            if doc_id not in docs_dict:
                docs_dict[doc_id] = {
                    'name': metadata['document_name'],
                    'chunk_count': 0
                }
            docs_dict[doc_id]['chunk_count'] += 1

        # Create Document objects
        documents = []
        for doc_id, info in docs_dict.items():
            documents.append(Document(
                id=doc_id,
                name=info['name'],
                path="",
                chunk_count=info['chunk_count'],
                word_count=0,
                added_date=""
            ))

        return documents

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        document_ids: List[str] = None
    ) -> List[Tuple[str, str, str, float]]:
        """
        Perform hybrid search combining semantic and BM25 keyword search.

        Returns list of (content, doc_name, doc_id, score) tuples.
        """
        # Semantic search
        query_embedding = self._embed_text([query], is_query=True)[0]

        where_clause = None
        if document_ids:
            if len(document_ids) == 1:
                where_clause = {"document_id": document_ids[0]}
            else:
                where_clause = {"document_id": {"$in": document_ids}}

        semantic_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more for reranking
            where=where_clause
        )

        # Build semantic scores dict
        semantic_scores = {}
        if semantic_results and semantic_results['documents'] and semantic_results['documents'][0]:
            for i, doc in enumerate(semantic_results['documents'][0]):
                metadata = semantic_results['metadatas'][0][i]
                distance = semantic_results['distances'][0][i] if semantic_results['distances'] else 0
                score = 1 - distance  # Convert distance to similarity

                key = (doc, metadata['document_name'], metadata['document_id'])
                semantic_scores[key] = score

        # BM25 keyword search (if available)
        bm25_scores = {}
        if self._bm25_index is not None and self._chunk_texts:
            query_tokens = query.lower().split()
            bm25_raw_scores = self._bm25_index.get_scores(query_tokens)

            # Normalize BM25 scores
            max_bm25 = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1

            all_data = self._collection.get()
            for i, (doc, score) in enumerate(zip(self._chunk_texts, bm25_raw_scores)):
                if score > 0:
                    metadata = all_data['metadatas'][i]
                    # Filter by document_ids if specified
                    if document_ids and metadata['document_id'] not in document_ids:
                        continue
                    key = (doc, metadata['document_name'], metadata['document_id'])
                    bm25_scores[key] = score / max_bm25

        # Combine scores
        all_keys = set(semantic_scores.keys()) | set(bm25_scores.keys())
        combined_results = []

        for key in all_keys:
            sem_score = semantic_scores.get(key, 0)
            bm25_score = bm25_scores.get(key, 0)

            # Weighted combination
            combined_score = (self.SEMANTIC_WEIGHT * sem_score) + (self.BM25_WEIGHT * bm25_score)

            combined_results.append((key[0], key[1], key[2], combined_score))

        # Sort by combined score
        combined_results.sort(key=lambda x: x[3], reverse=True)

        return combined_results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = None,
        document_ids: List[str] = None
    ) -> List[RetrievedChunk]:
        """
        Search for relevant chunks using hybrid search.
        """
        self._ensure_initialized()

        if top_k is None:
            top_k = self.TOP_K

        results = self._hybrid_search(query, top_k, document_ids)

        chunks = []
        for content, doc_name, doc_id, score in results:
            chunks.append(RetrievedChunk(
                content=content,
                document_name=doc_name,
                document_id=doc_id,
                score=score
            ))

        return chunks

    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 3000,
        document_ids: List[str] = None
    ) -> str:
        """Get formatted context string for a query."""
        chunks = self.search(query, top_k=self.TOP_K, document_ids=document_ids)

        if not chunks:
            return ""

        # Build context string with relevance info
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate

        for chunk in chunks:
            chunk_text = chunk.content
            if total_chars + len(chunk_text) > max_chars:
                break

            # Include source and relevance
            relevance = f"{chunk.score:.0%}" if chunk.score else ""
            context_parts.append(
                f"[Source: {chunk.document_name} | Relevance: {relevance}]\n{chunk_text}"
            )
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    def clear_all(self):
        """Clear all documents from the database."""
        self._ensure_initialized()

        self._client.delete_collection("documents_v2")
        self._collection = self._client.get_or_create_collection(
            name="documents_v2",
            metadata={"hnsw:space": "cosine"}
        )
        self._bm25_index = None
        self._chunk_texts = []

    @property
    def document_count(self) -> int:
        """Get number of stored documents."""
        return len(self.get_documents())

    @property
    def is_empty(self) -> bool:
        """Check if database is empty."""
        return self.document_count == 0


def check_rag_dependencies() -> tuple:
    """Check if RAG dependencies are installed."""
    missing = []

    try:
        import chromadb
    except ImportError:
        missing.append("chromadb")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")

    # rank_bm25 is optional but recommended
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        pass  # Optional

    if missing:
        return False, f"Missing: {', '.join(missing)}\nInstall with: pip install {' '.join(missing)}"

    return True, ""
