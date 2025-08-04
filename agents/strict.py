from typing import Dict

def grade_strict(question: str, answer: str, context: str) -> Dict:
    # Example rubric-matching agent
    score = 0
    explanation = ""
    if any(keyword in answer.lower() for keyword in context.lower().split()):
        score = 1
        explanation = "Answer matches rubric-relevant keywords."
    else:
        explanation = "Answer does not contain rubric keywords."
    return {"agent": "strict", "score": score, "explanation": explanation}

