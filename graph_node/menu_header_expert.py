"""
Menu Header Expert Node - Retrieves restaurant and chef information from menu headers.
"""
from langchain_core.messages import HumanMessage

from utils.boolean_query import BooleanQueryParser, boolean_searcher
from .config import llm, doc_splits, load_prompt


def ask_to_menu_header_expert(state):
    """
    Retrieve restaurant and chef information from menu headers.
    Uses boolean search to filter restaurants and chefs based on keywords.
    """
    print("---RETRIEVING FROM MENU Headers---")
    question = state["question"]
    keywords = state["keywords_extractor_answer"]

    try:
        if keywords["ristoranti"] or keywords["chef"] or keywords["pianeti"] or keywords["licenze chef"]:

            selected_keywords = []
            selected_keywords.extend(keywords["ristoranti"])
            selected_keywords.extend(keywords["chef"])
            selected_keywords.extend(keywords["licenze chef"])

            if state["planet_distance_answer"]:
                selected_keywords.extend(state["planet_distance_answer"])
            else:
                selected_keywords.extend(keywords["pianeti"])

            print(selected_keywords)

            if not selected_keywords:
                return {"menu_header_answer": ""}

            prompt = load_prompt(
                "boolean_query_prompt.txt",
                question=question,
                selected_keywords=selected_keywords,
            )

            response = llm.invoke(prompt)

            start = response.content.find("[")
            end = response.content.rfind("]")

            boolean_query = response.content[start:end + 1]
            print("Boolean query", boolean_query)
            parser = BooleanQueryParser(boolean_query)
            parsed_tree = parser.parse()
            print("Parsed query", parsed_tree)

            dict_menu_headers = boolean_searcher(parsed_tree, doc_splits, lower=False, header=True)

            print(dict_menu_headers)
            if not dict_menu_headers:
                return {"menu_header_answer": [-1]}

            if keywords["licenze chef"]:
                prompt = load_prompt(
                    "header_expert_prompt.txt",
                    question=question,
                    dict_menu_headers=dict_menu_headers,
                )
                response = llm.invoke(prompt)
                start = response.content.find("$")
                end = response.content.rfind("$")
                id_resturant = eval(response.content[start + 1:end])
                print("id_resturant", id_resturant)
                return {"menu_header_answer": id_resturant}

            return {"menu_header_answer": dict_menu_headers.keys()}
    except Exception as e:
        print("Header menu node error", e)

    return {"menu_header_answer": ""}

