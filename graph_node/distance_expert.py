"""
Distance Expert Node - Retrieves distance information between planets.
"""
from langchain_core.messages import HumanMessage

from .config import llm, dist_file, load_prompt


def ask_to_distance_expert(state):
    """
    Retrieve planet distance information from the distanze.txt file.
    Filters planets based on distance criteria in the question.
    """
    print("---RETRIEVING FROM DISTANCE---")
    question = state["question"]
    keywords = state["keywords_extractor_answer"]

    try:
        if "anni luce" in question:

            selected_keywords = []
            selected_keywords.extend(keywords["pianeti"])

            if not selected_keywords:
                return {"planet_distance_answer": []}

            prompt = load_prompt(
                "prompt/distance_expert_prompt.txt",
                question=question,
                dist_file=dist_file,
            )

            response = llm.invoke(prompt)

            start = response.content.find("[")
            end = response.content.rfind("]")
            planet_ok = eval(response.content[start:end + 1])
            print("Planet match: ", planet_ok)

            if planet_ok:
                prompt = load_prompt(
                    "prompt/distance_expert_prompt.txt",
                    question=question,
                    planet_ok=planet_ok,
                )
                response = llm.invoke(prompt)
                start = response.content.find('"')
                end = response.content.rfind('"')
                question = response.content[start + 1:end]
                print("Question", question)

            return {"planet_distance_answer": planet_ok, "question": question}
    except Exception as e:
        print("Distance node error", e)

    return {"planet_distance_answer": []}

