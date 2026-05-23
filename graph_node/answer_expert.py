"""
Answer Expert Node - Generates the final answer by extracting dishes.
"""

import re


def generate(state):
    """
    Generate the final answer by extracting dish information.
    Extracts dishes from the menu expert answer.
    Returns empty list if more than 30 dishes are found.
    """
    print("---GENERATING ANSWER---")
    menu_expert_answer = state["menu_expert_answer"]

    dishes = set()
    for x in menu_expert_answer:
        if isinstance(menu_expert_answer[x], str):
            matches = re.findall(r"<dish>(.*?)</dish>", menu_expert_answer[x])
            for match in matches:
                dishes.add(match)
        else:
            for y in menu_expert_answer[x]:
                matches = re.findall(r"<dish>(.*?)</dish>", y)

                for match in matches:
                    dishes.add(match)

    if len(dishes) > 30:
        return {"generation": {'dishes': []}}
    return {"generation": {'dishes': list(dishes)}}

