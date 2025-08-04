import pandas as pd
from agents.strict import grade_strict
from agents.semantic import grade_semantic
from retriever.faiss_retriever import FAISSRetriever
from confidence.entropy import compute_confidence
from feedback.logger import log_feedback
from feedback.updater import compute_agent_performance, rewrite_main
import os
import json
from random import seed
from agents.learned import grade_learned,train_learned_agent

train_learned_agent()

# Seed for reproducibility
seed(42)

# Load data
all_data = pd.read_csv("data/asap-sas/train.tsv", sep="\t")
question_ids = all_data['EssaySet'].dropna().unique()[:2]
samples = []

# Prepare 2 questions × 5 samples
for qid in question_ids:
    subset = all_data[all_data['EssaySet'] == qid]
    rubric_docs = subset['EssayText'].tolist()
    for _ in range(5):
        row = subset.sample(1).iloc[0]
        question = f"EssaySet {qid}"
        answer = row['EssayText']
        score = (row['Score1'] + row['Score2']) / 2
        samples.append((question, answer, score, rubric_docs))

agent_list = ["strict", "semantic", "learned"]

weights_path = "feedback/agent_weights.json"
if not os.path.exists(weights_path) or os.path.getsize(weights_path) == 0:
    with open(weights_path, "w") as f:
        json.dump({"weights": [1.0, 1.0]}, f)

for i, (question, answer, true_score, rubric_docs) in enumerate(samples):
    retriever = FAISSRetriever(rubric_docs)
    context = " ".join(retriever.retrieve(question + answer))

    outputs = []
    for agent in [grade_strict, grade_semantic]:
        outputs.append(agent(question, answer, context))
    outputs.append(grade_learned(question, answer, context, outputs))

    scores = [o['score'] for o in outputs]
    # Rescale scores to match the 0–3 rubric
    scores = [min(max(s * 3, 0), 3) for s in scores]
    for idx in range(len(outputs)):
        outputs[idx]["score"] = scores[idx]

    confidence = compute_confidence(scores)
    os.environ["DYNAMIC_CAP"] = str(confidence * 10)

    # Load adaptive weights if available
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
        "confidence": confidence,
        "true_score": true_score
    }

    log_feedback(result)
    logs = [result]
    agent_perf = compute_agent_performance(logs)
    weights = [agent_perf.get(a, 1.0) for a in agent_list]
    rewrite_main(weights, agent_perf)
    print(f"✅ Graded sample {i+1}/10")

from feedback.plot_performance import load_update_log, plot_agent_performance
plot_agent_performance(load_update_log())
