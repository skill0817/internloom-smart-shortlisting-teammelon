"""
semantic_matcher.py
===================
Calculates the semantic similarity between a Job Description and a Candidate's
Resume using sentence embeddings and cosine similarity.
WHAT ARE EMBEDDINGS?
--------------------
Computers cannot understand raw words or text directly.
An "Embedding" converts a piece of text into a list of numbers (a vector) in a
high-dimensional mathematical space (e.g. 384 dimensions for all-MiniLM-L6-v2).
Text with similar meanings end up closer together in this mathematical space,
even if they use different words (e.g. "Software Engineer" and "Developer").
WHAT IS COSINE SIMILARITY?
--------------------------
Cosine similarity measures the angle between two vectors:
- If two vectors point in the exact same direction, cosine similarity is 1.0 (100% match).
- If they are completely orthogonal/unrelated, it is close to 0.0.
- If they point in opposite directions, it is negative.
By calculating the cosine similarity of the Job Description embedding and the
Resume embedding, we get an objective, deterministic measurement of how well
the candidate's experience matches the job context.
"""
from typing import Dict, Any, Optional
import numpy as np
# Global cache for the embedding model so it only loads once into memory.
# Loading a model takes a few seconds; caching keeps subsequent candidate matches instant.
_CACHED_MODEL = None
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """
    Loads and returns the SentenceTransformer model.
    Uses a singleton pattern (caching) so the model is not re-downloaded
    or re-initialized repeatedly.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        # Import lazily so importing other modules doesn't have startup delay
        from sentence_transformers import SentenceTransformer
        _CACHED_MODEL = SentenceTransformer(model_name)
    return _CACHED_MODEL

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates cosine similarity between two 1-D vectors:
    
        cosine_similarity = (A . B) / (||A|| * ||B||)
        

    where:
    - (A . B) is the dot product of A and B
    - ||A|| and ||B|| are the Euclidean magnitudes (norms) of vectors A and B
    
    Returns a float typically between 0.0 and 1.0.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot_product = np.dot(vec_a, vec_b)
    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)
def calculate_semantic_score(
    job_description_text: str,
    resume_text: str,
    model=None
) -> Dict[str, float]:
    """
    Computes semantic similarity between job description text and resume text.
    
    Steps:
    1. Validate input text (handles empty/blank strings safely).
    2. Convert Job Description into an embedding vector.
    3. Convert Resume into an embedding vector.
    4. Compute cosine similarity between the two vectors.
    5. Convert similarity to a 0.0 to 100.0 score.
    
    Parameters:
        job_description_text (str): The full job description text.
        resume_text (str): The full resume text.
        model (SentenceTransformer, optional): Pass an existing model instance to reuse it.
        
    Returns:
        dict: {"semantic_score": float between 0.0 and 100.0}
 
    Parameters:
        job_description_text (str): The full job description text.
        resume_text (str): The full resume text.
        model (SentenceTransformer, optional): Pass an existing model instance to reuse it.
        
    Returns:
        dict: {"semantic_score": float between 0.0 and 100.0}
    """
    # Defensive check: if either text is empty or blank, similarity is 0.0
    if not isinstance(job_description_text, str) or not job_description_text.strip():
        return {"semantic_score": 0.0}
    if not isinstance(resume_text, str) or not resume_text.strip():
        return {"semantic_score": 0.0}
    # Load model if not provided
    if model is None:

        model = get_embedding_model()
    # Step 1: Generate embeddings for both texts
    # sentence-transformers outputs normalized numpy vectors when normalize_embeddings=True
    jd_embedding = model.encode(job_description_text, convert_to_numpy=True)
    resume_embedding = model.encode(resume_text, convert_to_numpy=True)
    # Step 2: Compute Cosine Similarity
    raw_similarity = compute_cosine_similarity(jd_embedding, resume_embedding)
    # Step 3: Scale to 0 - 100 range
    # Clamp negative values to 0.0 (unrelated text shouldn't produce negative scores)
    clamped_similarity = max(0.0, min(1.0, raw_similarity))
    semantic_score = round(clamped_similarity * 100.0, 1)
    return {
        "semantic_score": semantic_score
    }
