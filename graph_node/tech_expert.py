"""
Tech Expert Node - Retrieves technical information from the technical documents.
"""

from langchain_core.messages import HumanMessage
from .config import llm, tech_docs


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

            response = llm.invoke([HumanMessage(content=formatted_rag_prompt)])
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

