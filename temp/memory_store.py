import json
import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer

class MemoryStore:
    """Store and retrieve memory of clip extraction performance and user feedback."""
    
    def __init__(self, memory_file: str = "clip_memory.json", embedding_model: str = "all-MiniLM-L6-v2"):
        self.memory_file = memory_file
        self.memory = self._load_memory()
        # Initialize embedding model for semantic similarity
        self.embedder = SentenceTransformer(embedding_model)
    
    def _load_memory(self) -> Dict:
        """Load memory from file or create new if not exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"prompts": {}, "metadata": {"created_at": datetime.now().isoformat()}}
        else:
            return {"prompts": {}, "metadata": {"created_at": datetime.now().isoformat()}}
    
    def _save_memory(self) -> None:
        """Save memory to file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def add_extraction(self, prompt: str, video_id: str, scenes: List[Dict], 
                      selected_scenes: List[Dict]) -> str:
        """Add a new extraction to memory."""
        # Generate a unique memory ID for this extraction
        memory_id = str(uuid.uuid4())
        
        # Create embedding for prompt for future similarity matching
        prompt_embedding = self.embedder.encode(prompt).tolist() if prompt else None
        
        # Store the extraction with timestamp
        self.memory["prompts"][memory_id] = {
            "prompt": prompt,
            "prompt_embedding": prompt_embedding,
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "scenes": scenes,
            "selected_scenes": selected_scenes,
            "feedback": None
        }
        
        self._save_memory()
        return memory_id
    
    def add_feedback(self, memory_id: str, is_accurate: bool, 
                   feedback_text: Optional[str] = None) -> bool:
        """Add user feedback for a specific extraction."""
        if memory_id not in self.memory["prompts"]:
            return False
        
        self.memory["prompts"][memory_id]["feedback"] = {
            "is_accurate": is_accurate,
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_memory()
        return True
    
    def find_similar_extractions(self, prompt: str, limit: int = 3) -> List[Dict]:
        """Find similar prompts and their extraction results."""
        if not prompt or not self.memory["prompts"]:
            return []
        
        # Create embedding for the current prompt
        prompt_embedding = self.embedder.encode(prompt).tolist()
        
        # Get all past prompts with embeddings
        past_prompts = []
        for memory_id, data in self.memory["prompts"].items():
            if data["prompt_embedding"] and data["feedback"] and data["feedback"]["is_accurate"]:
                embedding = data["prompt_embedding"]
                past_prompts.append({
                    "memory_id": memory_id,
                    "embedding": embedding,
                    "data": data
                })
        
        if not past_prompts:
            return []
        
        # Calculate similarity with past prompts
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarities = []
        for past in past_prompts:
            similarity = cosine_similarity(
                [prompt_embedding], 
                [past["embedding"]]
            )[0][0]
            similarities.append((past["memory_id"], similarity, past["data"]))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N most similar
        return [
            {
                "memory_id": s[0],
                "similarity": s[1],
                "prompt": s[2]["prompt"],
                "selected_scenes": s[2]["selected_scenes"],
                "feedback": s[2]["feedback"]
            } 
            for s in similarities[:limit]
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory performance."""
        total = len(self.memory["prompts"])
        if total == 0:
            return {
                "total_extractions": 0,
                "feedback_received": 0,
                "accurate_percentage": 0,
                "inaccurate_percentage": 0
            }
        
        with_feedback = 0
        accurate = 0
        
        for memory_id, data in self.memory["prompts"].items():
            if data["feedback"]:
                with_feedback += 1
                if data["feedback"]["is_accurate"]:
                    accurate += 1
        
        return {
            "total_extractions": total,
            "feedback_received": with_feedback,
            "accurate_percentage": (accurate / with_feedback * 100) if with_feedback > 0 else 0,
            "inaccurate_percentage": ((with_feedback - accurate) / with_feedback * 100) if with_feedback > 0 else 0
        }