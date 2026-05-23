"""
Distance Expert Node - Retrieves distance information between planets.
"""

from .config import llm, dist_file


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

            prompt = f"""
            Hai a disposizione:
                Un testo contenente tutte le distanze tra i pianeti della galassia.
                Una query in linguaggio naturale.

            Obiettivo:
                Dopo aver analizzato il testo delle distanze trova i pianeti che rispondono alla domanda.
                Ritorna solo i pianeti in una lista python.
                
                
            Input:
                Query: "{question}"
                Testo distanze: "{dist_file}"
                
            
            Esempio: Query:"Quali pianeti sono in un raggio di 134 anni luce dal pianeta Urano?"
                     Output:["Pianeta 1", "Pianeta 2"]
                     
            Attenzione: se nella domanda c'e' scritto di includere il pianeta di partenza ricordati di inserirlo nella risposta.
                
            """

            response = llm.invoke(prompt)

            start = response.content.find("[")
            end = response.content.rfind("]")
            planet_ok = eval(response.content[start:end + 1])
            print("Planet match: ", planet_ok)

            if planet_ok:
                response = llm.invoke(f"""Riscrivi la seguente query escludendo la parte relativa alle distanze e aggiungi, al suo interno, la risposta con le distanze dei pianeti. Query: '{question}' Lista dei pianeti da integrare: {planet_ok}""")
                start = response.content.find('"')
                end = response.content.rfind('"')
                question = response.content[start + 1:end]
                print("Question", question)

            return {"planet_distance_answer": planet_ok, "question": question}
    except Exception as e:
        print("Distance node error", e)

    return {"planet_distance_answer": []}

