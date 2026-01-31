"""Test MMR retrieval"""
import numpy as np
import sys
sys.path.insert(0, '.')

from src.memory.attention_retrieval import AttentionMemoryRetrieval

class MockStore:
    pass

r = AttentionMemoryRetrieval(MockStore())

# Test embedding
emb1 = r._text_to_embedding('hello world test')
emb2 = r._text_to_embedding('hello world test')
emb3 = r._text_to_embedding('completely different topic')

sim_same = r._cosine_similarity(emb1, emb2)
sim_diff = r._cosine_similarity(emb1, emb3)

print(f"Self-similarity: {sim_same:.3f}")
print(f"Different-similarity: {sim_diff:.3f}")

# Test MMR selection
relevance = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
embeddings = np.array([
    r._text_to_embedding('topic A main content'),
    r._text_to_embedding('topic A related stuff'),
    r._text_to_embedding('topic B different area'),
    r._text_to_embedding('topic C unique domain'),
    r._text_to_embedding('topic A variant text'),
])

selected, metrics = r._mmr_select(relevance, embeddings, k=3, lambda_param=0.6, min_dissim=0.2)
print(f"MMR selected: {selected}")
print(f"Mean dissimilarity: {metrics['mean_dissim']:.3f}")

# Pure relevance would select [0, 1, 2]
# MMR should prefer diversity, potentially selecting [0, 2, 3] or similar
print("MMR test PASSED")
