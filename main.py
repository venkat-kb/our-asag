from agents.strict import grade_strict
from agents.semantic import grade_semantic
from retriever.faiss_retriever import FAISSRetriever
from confidence.entropy import compute_confidence
from feedback.logger import log_feedback

# Sample context/rubric
docs = [
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Mitochondria are the powerhouse of the cell.",
    "Gravity causes objects to fall toward Earth."
]
retriever = FAISSRetriever(docs)

question = "What does photosynthesis do?"
answer = "It converts sunlight to chemical energy."

context = " ".join(retriever.retrieve(question + answer))

outputs = []
for agent in [grade_strict, grade_semantic]:
    outputs.append(agent(question, answer, context))

scores = [o['score'] for o in outputs]
confidence = compute_confidence(scores)
final_score = sum(scores) / len(scores)

result = {
    "question": question,
    "answer": answer,
    "context": context,
    "agents": outputs,
    "final_score": final_score,
    "confidence": confidence
}

log_feedback(result)
print(result)
