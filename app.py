import streamlit as st
import os
import re
import logging # Use logging  a feature of streamlit
from transcription import get_transcript
from rag import (
    initialize_rag,
    generate_response,
    list_vector_dbs,
    delete_vector_db,
    hybrid_retrieve,
    configure_gemini_api,
    VECTOR_DB_DIR,
    embedding_model
)


st.set_page_config(layout="wide")
#### Just confirming everything is working or not

try:
    from transcription import get_transcript
except ImportError:
    st.error("⚠️ `transcription.py` not found. Please ensure it exists and is importable.")
    def get_transcript(url: str) -> str:
        st.warning("Using dummy transcript function.")
        return f"Dummy transcript for URL: {url}. Please provide the real `transcription.py`."


# Initializing Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if 'gemini_configured' not in st.session_state:
    st.session_state.gemini_configured = False
# End of Session State Initialization


#initial layot
st.title("🎮 Chat with YouTube Video Transcript")

# Configure Gemini API using Streamlit Secrets - Run once per session start/refresh
# Use session state to avoid re-configuring unnecessarily on every script rerun
if not st.session_state.gemini_configured: 
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("🚨 **Configuration Error:** `GEMINI_API_KEY` not found in Streamlit Secrets (`.streamlit/secrets.toml`). Please add it.")
            logging.error("GEMINI_API_KEY not found in st.secrets")
        else:
            st.session_state.gemini_configured = configure_gemini_api(gemini_api_key)
            if not st.session_state.gemini_configured:
                st.error("🚨 **Configuration Error:** Failed to configure Gemini API using the key from Streamlit Secrets. Check the key's validity and logs.")
            else:
                logging.info("Gemini API successfully configured via Streamlit Secrets.")
                st.toast("Gemini AI Ready!", icon="✨") # Optional success message

    except Exception as e:
        # Handle cases where st.secrets might not be available or other errors occur
        st.error(f"🚨 Configuration Error:An unexpected issue occurred during secrets loading or Gemini configuration: {e}")
        logging.error(f"Error during initial Gemini configuration: {e}", exc_info=True)
        st.session_state.gemini_configured = False

# Check if Embedding model loaded correctly (it's loaded globally in rag_chroma)
if embedding_model is None:
    st.error("🚨 Initialization Error: The Sentence Transformer embedding model failed to load. RAG functionality will be unavailable. Check logs.")

# SIDEBAR FOR SETTINGS AND HISTORY
with st.sidebar:
    st.title("⚙️ History & Tuning")

    # RAG Retrieval Parameters
    st.subheader("Retrieval Settings")
    top_k = st.slider("Relevant Chunks to LLM (K)", min_value=1, max_value=10, value=4, step=1,
                        help="Maximum number of text chunks to retrieve as context.")
    score_threshold = st.slider("Relevance Threshold (Distance)", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                                    help="Maximum allowed distance score for chunks (Lower score = more similar/relevant). Stricter filter.")
    lambda_mult = st.slider("Diversity (MMR Lambda)", min_value=0.0, max_value=1.0, value=0.6, step=0.1,
                                    help="Balance relevance vs. diversity (0=Max Relevance, 1=Max Diversity).")
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=100,
                            help="Size of text chunks to be indexed. Smaller chunks = more context but slower retrieval.")
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=100, value=0, step=10,
                                help="Number of overlapping tokens between chunks. More overlap = better context but slower retrieval.")
    # Vector Store Management
    st.subheader("🗑️ Manage Embeddings(VectorDb)")
    # ... (vector store management remains the same) ...
    saved_video_ids = list_vector_dbs()
    if saved_video_ids:
        selected_vid_to_delete = st.selectbox("Select video ID to delete:", saved_video_ids, key="delete_select")
        if st.button("Delete Selected Video Embeddings", key="delete_single"):
            if selected_vid_to_delete:
                with st.spinner(f"Deleting embeddings for {selected_vid_to_delete}..."):
                    delete_vector_db(selected_vid_to_delete)
                st.success(f"Deleted embeddings for video: {selected_vid_to_delete}")
                # Force rerun to update the selectbox
                st.rerun()
            else:
                st.warning("Please select a video ID to delete.")

        if st.button("⚠️ Delete All Embeddings", key="delete_all"):
            # Use a secondary confirmation for safety
            if st.checkbox("Confirm deletion of ALL stored embeddings?", key="confirm_delete_all"):
                with st.spinner("Deleting all embeddings..."):
                    current_ids = list_vector_dbs() # Get list again before iterating
                    for vid in current_ids:
                        delete_vector_db(vid)
                st.success("All stored embeddings have been deleted.")
                st.rerun() # Force rerun to update UI
            else:
                st.warning("Deletion not confirmed.")
    else:
        st.info("No embeddings saved yet.")


    # Chat History Display Area
    st.subheader("📜 Chat History")
    chat_history_placeholder = st.empty()


# Main Page Logic
# Only proceed if Gemini is configured and embedding model loaded
APP_READY = st.session_state.get('gemini_configured', False) and (embedding_model is not None)

if not APP_READY:
    st.warning("Application is not ready. Please check configuration errors above or in the logs.")
else:
    # Input for YouTube URL
    video_url = st.text_input("YouTube Video URL:")

    # Regex for Video ID
    VIDEO_ID_REGEX = r"(?:v=|youtu\.be\/|embed\/|watch\?v=|&v=|\/v\/)([\w-]{11})(?:\?|&|$)"

    def setup_rag_for_video(url):
        match = re.search(VIDEO_ID_REGEX, url)
        if not match:
            st.error("Invalid YouTube URL or could not extract Video ID.")
            st.session_state.current_video_id = None
            st.session_state.vectordb = None
            return None, None # Return None for video_id and vectordb

        video_id = match.group(1)
        logging.info(f"Extracted Video ID: {video_id}")

        # Check if we're already working with this video
        if st.session_state.current_video_id == video_id and st.session_state.vectordb:
            logging.info(f"Using RAG for video ID: {video_id}")
            return video_id, st.session_state.vectordb

        # New video ID or no RAG loaded
        st.session_state.messages = [] # Clear chat history for new video
        st.session_state.current_video_id = video_id
        st.session_state.vectordb = None # Reset vectordb state

        try:
            #Getting Transcript
            with st.spinner(f"Fetching transcript..."):
                transcript_text = get_transcript(url) 
                if not transcript_text or transcript_text.startswith("Dummy transcript"):
                    st.warning(f"Could not fetch a valid transcript for {url}.")
                    st.error("Failed to get transcript. Cannot proceed with chat.")
                    st.session_state.current_video_id = None
                    return None, None

            
            with st.expander("View Transcript Snippet"):
                st.text(transcript_text[:500] + "...") # Show a snippet

            # 2. Initialize RAG (Vector Store)
            with st.spinner(f"Initializing RAG..."):
                vectordb = initialize_rag(transcript_text, video_id, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            if vectordb is None:
                st.error(f"Failed to initialize RAG system for video {video_id}. Please check logs or try again.")
                st.session_state.current_video_id = None
                return None, None



            # Display total chunks info in sidebar (after successful init)
            try:
                # Ensure vectordb is valid before calling get()
                if hasattr(vectordb, 'get'):
                    texts = vectordb.get()['documents'] # Get documents from loaded DB
                    st.sidebar.markdown(f"🔢 **Total Chunks Indexed:** {len(texts)}")
                else:
                    raise ValueError("Vectordb object is invalid or None.")
            except Exception as e:
                logging.warning(f"Could not retrieve chunk count from vectordb: {e}")
                st.sidebar.markdown("🔢 **Total Chunks Indexed:** (Unavailable)")

            return video_id, vectordb

        except Exception as e:
            st.error(f"⚠️ An error occurred during setup for {url}: {str(e)}")
            logging.error(f"Setup failed for URL {url}", exc_info=True)
            st.session_state.current_video_id = None
            st.session_state.vectordb = None
            return None, None


    # Main Interaction Logic
    if video_url:
        current_video_id, current_vectordb = setup_rag_for_video(video_url)

        if current_video_id and current_vectordb:
            # Display existing chat messages
            # chat message display loop remains the same
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])


            # Update sidebar chat history display
            # sidebar history update remains the same
            with chat_history_placeholder.container():
                if st.session_state.messages:
                    for msg in st.session_state.messages:
                        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content'][:100]}...") # Show snippet
                else:
                    st.write("Chat history will appear here.")


            # Get user input
            if user_query := st.chat_input(f"Ask about video..."):
                # user message
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                # Retrieve and Generate
                with st.chat_message("Carl Sagan"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking... 🤔")

                    # Perform retrieval
                    with st.spinner("Finding relevant parts..."):
                        try:
                            relevant_chunks, scores = hybrid_retrieve(
                                current_vectordb, user_query,
                                top_k=top_k, score_threshold=score_threshold, lambda_mult=lambda_mult
                            )
                            logging.info(f"Retrieved {len(relevant_chunks)} chunks.")
                            # ... (display retrieved chunks in sidebar remains the same) ...
                            with st.sidebar.expander("📚 Retrieved Context Chunks", expanded=False):
                                if relevant_chunks:
                                    for i, (chunk, score) in enumerate(zip(relevant_chunks, scores), start=1):
                                        st.markdown(f"**Chunk {i}** (Score: `{score:.4f}`):")
                                        st.caption(f"{chunk[:200]}...") # Show snippet
                                else:
                                    st.write("No relevant chunks passed the retrieval criteria.")

                        except Exception as retrieve_err:
                            st.error(f"Error during retrieval: {retrieve_err}")
                            logging.error("Retrieval failed.", exc_info=True)
                            relevant_chunks = []

                    # Generate response
                    with st.spinner("Crafting response..."):
                        try:
                            response_text = generate_response(relevant_chunks, user_query)
                        
                        except Exception as gen_err:
                            # generate_response now includes specific error handling for auth
                            st.error(f"Error during response generation: {gen_err}") # Display error in chat
                            logging.error("Generation failed.", exc_info=True)
                            response_text = f"⚠️ Apologies, failed to generate response: {gen_err}"


                    # Display final response
                    message_placeholder.markdown(response_text)

                # Add assistant response to state
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Update sidebar history again
                # ... (sidebar history update remains the same) ...
                with chat_history_placeholder.container():
                    for msg in st.session_state.messages:
                        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content'][:100]}...") # Show snippet


    else: # No URL entered yet
        if st.session_state.current_video_id: # Clear state if a previous video was loaded
            st.session_state.messages = []
            st.session_state.current_video_id = None
            st.session_state.vectordb = None
            st.rerun() # Rerun to clear the display associated with the old video
        #st.info("Please enter a YouTube video URL to begin.")

