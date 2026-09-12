"""
scorer.py
=========
Combines the Keyword Score and Semantic Score into a single, balanced Final Score.
FORMULA:
    FINAL SCORE = (SEMANTIC_WEIGHT * semantic_score) + (KEYWORD_WEIGHT * keyword_score)
Default:
    60% Semantic Match + 40% Keyword Match
"""
from typing import Dict, Any
# ==============================================================================
# SCORING WEIGHT CONFIGURATION
# Adjust these constants to tune the balance between semantic and keyword scoring.
# Together they should typically sum to 1.0 (100%).
# ==============================================================================
SEMANTIC_WEIGHT = 0.60  # 60% weight on contextual / conceptual understanding
KEYWORD_WEIGHT = 0.40   # 40% weight on explicit technical skill presence
def calculate_final_score(
    keyword_score: float,
    semantic_score: float,
    semantic_weight: float = SEMANTIC_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT
) -> Dict[str, float]:
    """
    Combines keyword_score and semantic_score into a final weighted score.
    
    Parameters:
        keyword_score (float): Score from 0.0 to 100.0 from keyword matching.
        semantic_score (float): Score from 0.0 to 100.0 from semantic similarity.
        semantic_weight (float): Multiplier for semantic score (default 0.60).
        keyword_weight (float): Multiplier for keyword score (default 0.40).
        
    Returns:
        dict:
            {
                "keyword_score": 80.0,
                "semantic_score": 85.0,
                "final_score": 83.0
            }
    """
    # Ensure scores are valid positive floats
