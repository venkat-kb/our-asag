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

# Load adaptive weights if available
import os, json
weights_path = "feedback/agent_weights.json"
if os.path.exists(weights_path):
    with open(weights_path, "r") as f:
        default_weights = json.load(f)["weights"]
else:
    default_weights = [1.0 for _ in outputs]

final_score = sum(w * s for w, s in zip(default_weights, scores)) / sum(default_weights)

result = {
    "question": question,
    "answer": answer,
    "context": context,
    "agents": outputs,
    "final_score": final_score,
    "confidence": confidence
}

log_feedback(result)

from feedback.updater import compute_agent_performance, rewrite_main
logs = [result]  # simulate live feedback
agent_perf = compute_agent_performance(logs)
agent_list = ["strict", "semantic"]
weights = [agent_perf.get(a, 1.0) for a in agent_list]
rewrite_main(weights, agent_perf)
print(result)

try:
    from feedback.plot_performance import load_update_log, plot_agent_performance
    plot_agent_performance(load_update_log())
except Exception as e:
    print("⚠️ Could not generate performance graph:", e)