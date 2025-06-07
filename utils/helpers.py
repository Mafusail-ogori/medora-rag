from typing import Dict, Any, Tuple
from fastapi import HTTPException


def get_top_probability(cnn_data: Dict[str, Any]) -> Tuple[str, float]:
    """Extract the top probability prediction from CNN data"""
    probabilities = {k: v for k, v in cnn_data.items() if k != 'image_url'}

    if not probabilities:
        raise HTTPException(status_code=400, detail="No probabilities found in CNN data")

    top_condition = max(probabilities.items(), key=lambda x: x[1])
    return top_condition[0], top_condition[1]
