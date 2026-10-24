
from answer_planner import create_answer_plan

query = "What does the abstract say about the proposed system?"

print("INPUT:", repr(query))

plan = create_answer_plan(query)

print("ORIGINAL:", repr(plan["original_question"]))
print("QUESTION:", repr(plan["question"]))
print("CLEAN:", repr(plan["clean_question"]))
print("INTENT:", plan["intent"])
