#library for vectore store,remove duplicates cosine simalrity and clustering
import os 
import logging
import shutil
import numpy as np
import hdbscan    #Group similar chunks (clustering)
from typing import List, Tuple, Any, Optional
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_distances


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.utils import maximal_marginal_relevance

import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VECTOR_DB_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gemini-1.5-flash"

# Load embedding model 
try:
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logging.info("Embedding model loaded successfully")
except Exception as e:
    logging.error("Failed to load embedding model", exc_info=True)
    embedding_model = None

#Gemini API key configuration
def configure_gemini_api(api_key: str) -> bool:
    if not api_key:
        logging.error("No API key provided")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        # Print the full error message to the terminal
        logging.error("Gemini API setup failed", exc_info=True)
        return False

#split the big transcript into many small chunks
def split_transcript(text: str, chunk_size: int = 700, chunk_overlap: int = 0) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

#normalize text by lowercasing and removing extra spaces
def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

#remove duplicates from the list of chunks
def remove_duplicates(chunks: List[str]) -> List[str]:
    unique = OrderedDict()
    for chunk in chunks:
        key = normalize(chunk)
        if key not in unique:
            unique[key] = chunk
    return list(unique.values())


def reset_vector_db(video_id: str) -> None:
    # Create the full path to the vector DB folder for the video
    path = os.path.join(VECTOR_DB_DIR, video_id)
    if os.path.exists(path):
        try:
            # Delete the folder and everything inside it
            shutil.rmtree(path)
        except OSError as e:
            logging.error(f"Failed to remove DB: {e}")


def delete_vector_db(video_id: str) -> None:
    reset_vector_db(video_id)
    pkl_path = os.path.join(VECTOR_DB_DIR, f"{video_id}.pkl")
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
        except OSError as e:
            logging.error(f"Failed to remove .pkl file: {e}")


def list_vector_dbs() -> List[str]:
    # If main vector store doesn't exist, return an empty list
    if not os.path.exists(VECTOR_DB_DIR):
        return []
    try:
        # Get a list of all folder names for showing in the UI
        return [f for f in os.listdir(VECTOR_DB_DIR) if os.path.isdir(os.path.join(VECTOR_DB_DIR, f))]
    except OSError as e:
        logging.error(f"Error listing DBs: {e}")
        return []


def initialize_rag(transcript_text: str, video_id: str,chunk_size,chunk_overlap) -> Optional[Chroma]:
    #stop if embedding model is not loaded
    if embedding_model is None:
        return None
    #folder path where the vector DB will be stored
    persist_path = os.path.join(VECTOR_DB_DIR, video_id)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True) # if vectordb exists already

    try:
        # Check if the directory already exists
        if os.path.exists(os.path.join(persist_path, "chroma-collections.parquet")):
            try:
                vectordb = Chroma(persist_directory=persist_path, embedding_function=embedding_model)
                _ = vectordb.get(limit=1)
                return vectordb
            except Exception as e:
                logging.warning(f"DB load failed, rebuilding: {e}")
                reset_vector_db(video_id)
        #if the directory doesn't exist, create it
        chunks = split_transcript(transcript_text,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = remove_duplicates(chunks)
        if not chunks:
            return None

        # Create a list of Document objects from the chunks
        docs = [Document(page_content=chunk) for chunk in chunks]
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_path
        )
        #saving to disk
        vectordb.persist()
        return vectordb
    #for errors
    except Exception as e:
        logging.error(f"RAG init failed: {e}", exc_info=True)
        reset_vector_db(video_id)
        return None


def _validate_embeddings(docs: List[Document], embeddings: List[Any]) -> Tuple[List[Document], List[np.ndarray]]:
    valid_docs = []
    valid_embeddings = []
    #making sure the length of docs and embeddings are same
    min_len = min(len(docs), len(embeddings))
    #looping through the docs and embeddings
    for doc, emb in zip(docs[:min_len], embeddings[:min_len]):
        # Check if the embedding is a list of floats or ints
        if isinstance(emb, list) and all(isinstance(x, (float, int)) for x in emb):
            #keeping the valid ones 
            valid_docs.append(doc)
            valid_embeddings.append(np.array(emb, dtype=np.float32))
    return valid_docs, valid_embeddings# return the filtered docs and embeddings


def cluster_and_select_indices(docs, embeddings, min_cluster_size=2):
    # Filter out any bad embeddings
    valid_docs, valid_embeddings = _validate_embeddings(docs, embeddings)

    # If no usable embeddings, just return the first few docs
    if not valid_embeddings:
        return list(range(min(len(docs), 5)))

    # If too few for clustering, skip it and return all valid ones
    if len(valid_embeddings) < min_cluster_size:
        return list(range(len(valid_docs)))

    try:
        # Combine all embeddings into one big matrix
        X = np.vstack(valid_embeddings)

        # Calculate distances between every pair of embeddings
        distance_matrix = cosine_distances(X)

        # Set the diagonal (self-to-self) distances to 0
        np.fill_diagonal(distance_matrix, 0)

        # clustering using HDBSCAN on the distance matrix
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(distance_matrix.astype(np.float64))

        # Group documents by cluster label
        clusters = {}
        noise_count = 0
        for i in range(len(labels)):
            label = labels[i]

            # Give noise points their own unique label
            if label == -1:
                cluster_label = "noise-" + str(noise_count)
                noise_count += 1
            else:
                cluster_label = label

            if cluster_label not in clusters:
                clusters[cluster_label] = []

            clusters[cluster_label].append(i)

        # From each cluster, pick the first document
        selected_indices = []
        for cluster_id in sorted(clusters.keys()):
            first_doc_index = clusters[cluster_id][0]
            selected_indices.append(first_doc_index)

        return selected_indices

    except Exception as error:
        # If clustering fails for any reason, return first few valid ones
        logging.error("Clustering failed: %s" % error)
        return list(range(min(len(valid_docs), 5)))


def hybrid_retrieve(vectordb, query, top_k=5, score_threshold=0.4, lambda_mult=0.7, max_tokens=1000):
    # Making sure the vector DB and embedding model are ready
    if embedding_model is None or not isinstance(vectordb, Chroma):
        return [], []

    try:
        # Step 1: Converting the user query into an embedding vector
        query_embedding = embedding_model.embed_query(query)

        # Step 2: initialising results from the vector DB having more than k chunks
        results = vectordb.similarity_search_with_relevance_scores(query, k=max(top_k * 5, 20))

        # Step 3: Filtering out bad matches 
        filtered_results = []
        for doc, score in results:
            if score <= score_threshold:
                filtered_results.append((doc, score))

        # If nothing passed the threshold, stop
        if not filtered_results:
            return [], []

        # Step 4: Separate docs, scores, and re-embed the texts
        filtered_docs = []
        filtered_scores = []
        for doc, score in filtered_results:
            filtered_docs.append(doc)
            filtered_scores.append(score)

        # Embed all filtered docs again
        texts_to_embed = [doc.page_content for doc in filtered_docs]
        filtered_embeddings = embedding_model.embed_documents(texts_to_embed)

        # Step 5: Clustering and selecting indices
        selected_indices = cluster_and_select_indices(filtered_docs, filtered_embeddings)

        unique_docs = [filtered_docs[i] for i in selected_indices]
        unique_embeddings = [filtered_embeddings[i] for i in selected_indices]
        unique_scores = [filtered_scores[i] for i in selected_indices]

        # Step 6: Appling Maximal Marginal Relevance to select top diverse and relevant chunks
        mmr_embeddings = [np.array(embedding, dtype=np.float32) for embedding in unique_embeddings]
        mmr_indices = maximal_marginal_relevance(
            np.array(query_embedding, dtype=np.float32),
            mmr_embeddings,
            lambda_mult=lambda_mult,
            k=min(top_k, len(unique_docs))
        )

        # Get the selected documents and scores based on MMR
        selected_docs = [unique_docs[i] for i in mmr_indices]
        selected_scores = [unique_scores[i] for i in mmr_indices]

        # Step 7: Keep adding chunks until the token limit is hit
        final_chunks = []
        final_scores = []
        total_tokens = 0

        for doc, score in zip(selected_docs, selected_scores):
            tokens_in_doc = len(doc.page_content.split())

            if total_tokens + tokens_in_doc <= max_tokens:
                final_chunks.append(doc.page_content)
                final_scores.append(score)
                total_tokens += tokens_in_doc
            else:
                break

        return final_chunks, final_scores

    except Exception as error:
        # Catch and log any unexpected error
        logging.error("Retrieval failed: %s" % error)
        return [], []



def generate_response(relevant_texts: List[str], user_query: str) -> str:
    if not relevant_texts:
        return "No relevant content found. Try a different question."

    context = "\n\n---\n\n".join(relevant_texts)
    prompt = f"""
You’re Carl Sagan if he were a chill classmate.
Use only the transcript (“our notebook”) to answer.
Explain ideas clearly, using simple words, real-world analogies, and a poetic tone. Keep answers short to medium-length.
If it’s not in the notes, just say: “Not in my notes, bro.”


Transcript:
{context}

Question:
{user_query}
"""

    try:
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(prompt)

        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'prompt_feedback'):
            return f"Content blocked: {response.prompt_feedback.block_reason}"
        else:
            return "Failed to generate response"
    except Exception as e:
        if "API_KEY" in str(e):
            return "API key error - check your configuration"
        return f"Error generating response: {e}"
