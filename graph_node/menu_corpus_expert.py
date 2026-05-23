"""
Menu Corpus Expert Node - Retrieves dish information from menu content.
"""

from utils.boolean_query import BooleanQueryParser, boolean_searcher
from .config import llm, doc_splits
from .menu_header_expert import prompt_boolean_search


def ask_to_menu_corpus_expert(state):
    """
    Retrieve dish information from menu content.
    Uses boolean search to filter dishes based on techniques and ingredients.
    Intersects results with menu header expert results.
    """
    print("---RETRIEVING FROM MENU CORPUS---")
    question = state["question"]
    keywords = state["keywords_extractor_answer"]

    try:

        if keywords["tecniche galattiche"] or keywords["ingredienti"]:
            selected_keywords = []
            selected_keywords.extend(keywords["tecniche galattiche"])
            selected_keywords.extend(keywords["ingredienti"])
            print(selected_keywords)

            prompt = prompt_boolean_search.format(question=question, selected_keywords=selected_keywords)
            response = llm.invoke(prompt)

            start = response.content.find("[")
            end = response.content.rfind("]")

            boolean_query = response.content[start:end + 1]
            print(f"llm response: {boolean_query}")

            parser = BooleanQueryParser(boolean_query)
            parsed_tree = parser.parse()
            print(f"parsed query: {parsed_tree}")

            dict_menu_corpus = boolean_searcher(parsed_tree, doc_splits)

            id_extracted_headers = state["menu_header_answer"]

            if dict_menu_corpus:
                if id_extracted_headers:
                    new_dict = {}
                    for id in dict_menu_corpus:  # intersection between two ids
                        if id in id_extracted_headers:
                            new_dict[id] = f"Ristorante:{id}\n" + "\n".join(dict_menu_corpus[id])
                    dict_menu_corpus = new_dict
                    return {"menu_expert_answer": dict_menu_corpus}

                return {"menu_expert_answer": dict_menu_corpus}
    except Exception as e:
        print("Corpus menu node error", e)

    return {"menu_expert_answer": {}}

