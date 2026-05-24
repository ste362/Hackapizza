"""
Keywords Extractor Node - Extracts and categorizes keywords from user questions.
"""

import json
from .config import llm, load_prompt


def extract_keywords(state):
    """
    Extract keywords from the user question and categorize them.

    Categories:
    - licenze chef: Chef licenses
    - licenze tech: Technical licenses
    - pianeti: Planets
    - ingredienti: Ingredients
    - tecniche galattiche: Galactic techniques (proper nouns)
    - tecniche comuni: Common techniques
    - chef: Chef names
    - ristoranti: Restaurant names
    - abilità: Abilities
    """
    print("---Extracting keywords---")
    question = state["question"]

    prompt = load_prompt(
        "keywords_extractor_prompt.txt",
        question=question
    )

    response = llm.invoke(prompt)

    start = response.content.find("{")
    end = response.content.rfind("}")

    keyword = json.loads(response.content[start:end + 1])

    for k in keyword:
        selected_keywords = keyword[k]
        if k != "tecniche comuni":
            selected_keywords = [token for token in selected_keywords if token[0].isupper()]
        keyword[k] = selected_keywords

    print(keyword)
    return {"keywords_extractor_answer": keyword}

