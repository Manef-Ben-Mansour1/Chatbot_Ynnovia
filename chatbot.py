from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any
from pydantic import BaseModel, Field


# ------------------------------
# 1️⃣ Charger le modèle Ollama
# ------------------------------
llm = ChatOllama(model="phi3:mini")


# ------------------------------
# 2️⃣ Schéma de sortie extraction
# ------------------------------
class ChatbotOutput(BaseModel):
    question: str = Field(description="La question originale de l'utilisateur sans modification.")
    entity: str = Field(description="L'entité principale identifiée (ex: 'commande', 'produit').")
    intent: str = Field(description="L'intention de l'utilisateur (ex: 'post', 'get', 'update', 'delete').")
    query_params: Dict[str, Any] = Field(description="Les paramètres extraits sous forme de dictionnaire.")


# ------------------------------
# 3️⃣ Prompt d'extraction
# ------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Vous êtes un assistant de chatbot qui analyse les requêtes utilisateur pour identifier :
    - l'intention (intent)
    - l'entité principale (entity)
    - les paramètres pertinents (query_params)

    Vous devez retourner un JSON strictement conforme au schéma :
    {schema}

    Intent possibles : 'post', 'get', 'update', 'delete'
    Entités possibles : 'commande', 'produit', 'client', 'RFQ'
     Extraire seulement les paramètres (id, status, prix, nom, numero) avec le bon type (ex: id = entier).
    + Vous devez générer STRICTEMENT un JSON **sans commentaires**, sans texte inutile, sans remarques.
    + Ne jamais écrire "// ..." dans le JSON. Les valeurs manquantes doivent être omises ou mises à null sans explication.
    génére Json et only json

    """),
    ("human", "{query}")
])

# ------------------------------
# 4️⃣ Chaîne d'extraction
# ------------------------------
parser = JsonOutputParser(pydantic_object=ChatbotOutput)
extraction_chain = prompt | llm | parser


# ------------------------------
# 5️⃣ Mapping intelligent des entités
# ------------------------------
def map_entity(entity: str) -> str:
    entity_mapping = {
        "commande": "commande",
        "commandes": "commande",
        "order": "commande",
        "orders": "commande",
        "produit": "produit",
        "produits": "produit",
        "product": "produit",
        "products": "produit",
        "client": "client",
        "clients": "client",
        "customer": "client",
        "customers": "client"
    }
    return entity_mapping.get(entity.lower(), entity.lower())


# ------------------------------
# 6️⃣ Mapping des query_params
# ------------------------------
def map_query_params(query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mappe les paramètres d'entrée comme 'prix', 'numero' et 'price' pour les uniformiser.
    """
    mapped_params = {}
    for key, value in query_params.items():
        # Mappe 'prix', 'price' vers 'prix'
        if key.lower().startswith("prix") or key.lower() == "price":
            mapped_params["prix"] = value
        # Mappe 'numéro de commande', 'commande numéro' vers 'numero'
        elif key.lower().startswith("numéro") or "commande" in key.lower():
            mapped_params["numero"] = value
        else:
            mapped_params[key] = value
    return mapped_params


# ------------------------------
# 7️⃣ Simulation backend améliorée
# ------------------------------
def simulated_backend_call(entity: str, intent: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simule un appel backend : retourne un payload brut avec un code de statut dynamique.
    """
    print(f"\n--- Appel Backend Simulé ---")
    print(f"Entité: {entity}")
    print(f"Intention: {intent}")
    print(f"Paramètres: {query_params}")
    print(f"--------------------------")

    # Mapper les paramètres d'entrée
    query_params = map_query_params(query_params)

    # Initialiser un payload de base
    payload = {}

    # Traiter les commandes
    if entity == "commande":
        if intent == "post":
            if "prix" not in query_params or "status" not in query_params:
                payload["status"] = 400
                payload["message"] = "Bad Request: Prix ou statut manquant"
            else:
                payload["status"] = 200
                payload["commande_id"] = query_params.get("id", "N/A")
                payload["status_commande"] = query_params.get("status", "inconnu")
                payload["prix"] = query_params.get("prix", 0)
                payload["produit"] = query_params.get("produit", "non spécifié")
                payload["currency"] = query_params.get("currency", "inconnu")
        elif intent == "get":
            commande_id = query_params.get("id") or query_params.get("numero")
            if not commande_id:
                payload["status"] = 404
                payload["message"] = "Commande non trouvée"
            else:
                payload["status"] = 200
                payload["details"] = f"Détails commande #{commande_id}"
                payload["status_commande"] = "en attente"
                payload["prix"] = 300
        elif intent == "update":
            if "id" not in query_params:
                payload["status"] = 400
                payload["message"] = "Bad Request: ID de commande manquant"
            else:
                payload["status"] = 200
                payload["message"] = f"Commande #{query_params.get('id', 'N/A')} mise à jour"
                payload["nouveau_statut"] = query_params.get("status", "inconnu")
        elif intent == "delete":
            if "id" not in query_params:
                payload["status"] = 400
                payload["message"] = "Bad Request: ID de commande manquant"
            else:
                payload["status"] = 200
                payload["message"] = f"Commande #{query_params.get('id', 'N/A')} supprimée"

    # Traiter les produits
    elif entity == "produit":
        if intent == "post":
            if "nom" not in query_params or "prix" not in query_params:
                payload["status"] = 400
                payload["message"] = "Bad Request: Nom ou prix manquant"
            else:
                payload["status"] = 200
                payload["produit_nom"] = query_params.get("nom", "non spécifié")
                payload["prix"] = query_params.get("prix", "non spécifié")
                payload["specifications"] = query_params.get("specifications", "aucune")
                payload["currency"] = query_params.get("currency", "inconnu")
        elif intent == "get":
            produit_nom = query_params.get("nom")
            if not produit_nom:
                payload["status"] = 404
                payload["message"] = "Produit non trouvé"
            else:
                payload["status"] = 200
                payload["produit_nom"] = produit_nom
                payload["prix"] = query_params.get("prix", "non spécifié")
                payload["specifications"] = query_params.get("specifications", "aucune")
                payload["currency"] = query_params.get("currency", "inconnu")
        elif intent == "update":
            if "nom" not in query_params:
                payload["status"] = 400
                payload["message"] = "Bad Request: Nom du produit manquant"
            else:
                payload["status"] = 200
                payload["message"] = f"Produit {query_params.get('nom', 'N/A')} mis à jour"
                payload["nouveau_prix"] = query_params.get("prix", "non spécifié")
        elif intent == "delete":
            if "nom" not in query_params:
                payload["status"] = 400
                payload["message"] = "Bad Request: Nom du produit manquant"
            else:
                payload["status"] = 200
                payload["message"] = f"Produit {query_params.get('nom', 'N/A')} supprimé"

    # Traiter les clients
    elif entity == "client":
        if intent == "get":
            client_id = query_params.get("id")
            if not client_id:
                payload["status"] = 404
                payload["message"] = "Client non trouvé"
            else:
                payload["status"] = 200
                payload["client_id"] = client_id
                payload["nom_client"] = query_params.get("nom", "non spécifié")
                payload["email_client"] = query_params.get("email", "non spécifié")
                payload["adresse_client"] = query_params.get("adresse", "non spécifiée")

    # Ajouter d'autres entités au besoin...

    # Si l'entité n'est pas trouvée
    else:
        payload["status"] = 404
        payload["message"] = "Entité inconnue ou non supportée"

    # Retourner le payload avec le code de statut et message approprié
    return payload


# ------------------------------
# 7️⃣ Nouveau prompt de reformulation
# ------------------------------
final_response_prompt = PromptTemplate.from_template("""
Tu es un assistant.
Voici la question originale : "{question}"
Voici la réponse brute du backend : "{backend_raw}"

Rédige une réponse finale claire et naturelle pour l'utilisateur, en une phrase.
Ne dit rien de technique et ne répéte pas la reponse du backend.
Ne t'éloigne pas de l'information fournie, reformule de manière claire et directe sans complexité en une seule phrase.
Tout doit être donné précisément sans utiliser de termes approximatifs comme 'environ', 'approximativement' ou 'environ'.
Une seule phrase doit etre générée.
""")


# ------------------------------
# 8️⃣ Fonction principale
# ------------------------------
def run_chatbot(user_question: str):
    print(f"\n❓ Question utilisateur : {user_question}")
    try:
        # ➜ Extraction
        parsed_output = extraction_chain.invoke({"query": user_question, "schema": parser.get_format_instructions()})
        print(f"\n✅ Extraction : {parsed_output}")

        # Vérification si l'extraction a échoué
        if parsed_output is None or 'query_params' not in parsed_output or parsed_output['query_params'] is None:
            raise ValueError("L'extraction a échoué. Vérifiez les données d'entrée.")

        # ➜ Mapping de l'entité
        parsed_output['entity'] = map_entity(parsed_output['entity'])

        # ➜ Appel backend simulé
        backend_payload = simulated_backend_call(
            parsed_output['entity'],
            parsed_output['intent'],
            parsed_output['query_params']
        )
        print(f"\n✅ Payload backend : {backend_payload}")

        # ➜ Reformulation finale par LLM
        final_response = llm.invoke(final_response_prompt.format(
            question=user_question,
            backend_raw=backend_payload
        ))

        # ➜ Résultat complet (Extrait le contenu du message)
        result = {
            "question": parsed_output['question'],
            "entity": parsed_output['entity'],
            "intent": parsed_output['intent'],
            "query_params": parsed_output['query_params'],
            "backend_response": backend_payload,
            "final_response": final_response.content  # Accéder au contenu du message
        }

        print(f"\n💬 Réponse finale : {result['final_response']}")
        return result

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return {"error": str(e)}


# ------------------------------
# 9️⃣ Tests
# ------------------------------
if __name__ == "__main__":
    run_chatbot("Créer une commande pour le produit X avec le statut 'en attente' pour la commande #5299 et un prix de 14950.")
    run_chatbot("Quelle est la commande 1234 ?")
    run_chatbot("Donne-moi les détails du produit 'Ordinateur Portable'.")
