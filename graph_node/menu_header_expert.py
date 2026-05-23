"""
Menu Header Expert Node - Retrieves restaurant and chef information from menu headers.
"""

from utils.boolean_query import BooleanQueryParser, boolean_searcher
from .config import llm, doc_splits



# Prompt per la conversione da query in linguaggio naturale a query booleana, utilizzando solo le keywords fornite.
prompt_boolean_search = """
Hai a disposizione:
    Una lista di keywords.
    Una query in linguaggio naturale.

Obiettivo:
    Converti la query in linguaggio naturale in una query booleana, utilizzando solamente le keywords fornite.

Input:
    Query: "{question}"
    Keywords: {selected_keywords}

Istruzioni:
    Usa esclusivamente le keywords presenti nella lista Keywords fornita nell'input.
    Costruisci la query booleana impiegando gli operatori AND, OR, NOT.
    Le parentesi () possono essere utilizzate per definire la priorità degli operatori.
    Ogni keyword deve essere racchiusa tra virgolette doppie "".
    Restituisci solamente la query booleana, racchiusa tra parentesi quadre [].
    Ricorda che le keyword usate nella query booleana devono essere scritte nello stesso modo di come sono scritte nella lista fornita.
    Usa correttamente le parentesi e i doppi apici ma non inserire altri caratteri speciali.


Esempio 1: 
    Query: "Quali sono i piatti cucinati su ristoranti di Pandora che contengono Alghe Fluo e Spinaci Radioattivi ma non vengono cucinati secondo la tecnica di Fusione a Freddo"
    Keywords: ["Alghe Fluo", "Spinaci Radioattivi", "Fusione a Freddo"]
    Output atteso: [("Alghe Fluo" AND "Spinaci Radioattivi" AND NOT "Fusione a Freddo")]

Esempio 2: 
    Query: "Quali sono i piatti cucinati su ristoranti di Pandora, evitando rigorosamente quelli cucinati con Alghe Fluo?"
    Keywords: ["Alghe Fluo"]
    Output atteso: [NOT "Alghe Fluo"]

Esempio 3: 
    Query: "Quali sono i piatti cucinati da chef con licenza YCD di grado 5 su ristoranti di Pandora, evitando rigorosamente quelli cucinati con Alghe Fluo?"
    Keywords: ["YCD","Pandora"]
    Output atteso: ["YCD" AND "Pandora"]

Esempio 4:
    Query: "Quali piatti sono preparati utilizzando almeno una tecnica di taglio e una di surgelamento, ma senza l'uso di Polvere di Crononite?

        Tecniche di taglio da integrare: 
        - taglio dimensionale a lame fotofilliche
        - affettamento a pulsazioni quantistiche


        Tecniche di surgelamento da integrare:
        - cryo-tessitura energetica polarizzata
        - congelamento bio-luminiscente sincronico"

    Keywords: ["Polvere di Crononite","taglio dimensionale a lame fotofilliche", "affettamento a pulsazioni quantistiche", "cryo-tessitura energetica polarizzata", "congelamento bio-luminiscente sincronico"]
    Output atteso: [(NOT "Polvere di Crononite") AND (("taglio dimensionale a lame fotofilliche" OR "affettamento a pulsazioni quantistiche") AND ("cryo-tessitura energetica polarizzata" OR "congelamento bio-luminiscente sincronico"))]
"""



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

            prompt = prompt_boolean_search.format(question=question, selected_keywords=selected_keywords)

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
                response = llm.invoke(f"""Dati in input una query e un dizionario nel formato ristorante_id -> [informazioni ristorante], restituisci una lista contenente esclusivamente gli ID dei ristoranti in cui i gradi delle licenze degli chef soddisfano i requisiti specificati nella query.

                                            Query: {question}
                                            Dizionario: {dict_menu_headers}
                                            
                                            Restituisci solo la lista degli ID tra il token $, ad esempio:  $[1, 5, 6]$ """)
                start = response.content.find("$")
                end = response.content.rfind("$")
                id_resturant = eval(response.content[start + 1:end])
                print("id_resturant", id_resturant)
                return {"menu_header_answer": id_resturant}

            return {"menu_header_answer": dict_menu_headers.keys()}
    except Exception as e:
        print("Header menu node error", e)

    return {"menu_header_answer": ""}

