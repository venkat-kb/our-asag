from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def grade_semantic(question: str, answer: str, context: str):
    scores = util.cos_sim(model.encode(answer), model.encode(context))
    return {
        "agent": "semantic",
        "score": float(scores[0][0]),
        "explanation": "Semantic similarity score based on context."
    }