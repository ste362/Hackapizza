"""
Tech Expert Node - Retrieves technical information from the technical documents.
"""

from langchain_core.messages import HumanMessage
from .config import llm, tech_docs, load_prompt


def ask_to_tech_expert(state):
    """
    Retrieve technical information from the Codice Galattico.
    Integrates licenses and techniques from the extracted keywords.
    """
    print("---RETRIEVING FROM TECH---")

    question = state["question"]
    keywords = state["keywords_extractor_answer"]

    try:
        if keywords["licenze tech"] or keywords["tecniche comuni"]:

            doc = tech_docs[0]

            prompt = load_prompt(
                "tech_expert_prompt.txt",
                context = doc,
                licenze_tech = str(keywords["licenze tech"]),
                tecniche_comuni = str(keywords["tecniche comuni"])
            )

            response = llm.invoke(prompt)
            start = response.content.find("[")
            end = response.content.rfind("]")

            tech_retrieved = eval(response.content[start:end + 1])

            print(tech_retrieved)

            if tech_retrieved:
                prompt = load_prompt(
                    "tech_expert_prompt_2.txt",
                    context=doc,
                    question=question,
                    tech_retrieved=tech_retrieved
                )
                response = llm.invoke(prompt)

                start = response.content.find('"')
                end = response.content.rfind('"')
                question = response.content[start + 1:end]
                print("Tech question:", question)

            keywords["tecniche galattiche"] = tech_retrieved

            return {"keywords": keywords, "question": question}
    except Exception as e:
        print("Tech node error:", e)

    return {}

