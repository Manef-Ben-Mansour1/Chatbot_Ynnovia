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

    RÈGLES STRICTES :
    - Intent possibles UNIQUEMENT : 'post', 'get', 'put', 'delete'
    - Entités possibles UNIQUEMENT : 'commande', 'produit', 'vendor', 'buyer', 'RFQ'
    - Paramètres autorisés UNIQUEMENT(il faut choisir à partir de cette liste tout depend de la question de l'utilisateur) : 
    pour une commande on a  :total, priority,reference, vendor_reference, vendor, buyer, order_deadline, activities, source_document, untaxed, total, billing_status(nothing to bill or waiting to bill), confirmation_date, expected_arrival,status (cancelled or purchase order ),payment_terms,payment (paid / not paid)
    pour les informations produit d'une commande : product_name, description, quantity, received, billed, unitprice, taxes, disc,  fiscal_position,  due_date, tax_excluded
    pour toutes les commandes : avg_order_value, purchased_last_7_days, lead_time_to_purchase, rfqs_sent_last_7_days

    # voici un exemple de cette schéma :
    # {{
    # "question": "Afficher les commandes urgentes du fournisseur TechSolutions",
    # "entity": "commande",
    # "intent": "get",
    # "query_params": {{
    #     "vendor": "TechSolutions",
    #     "priority": "urgent"
    # }}
    # }}
    IMPORTANT :
    - Générez UNIQUEMENT un JSON valide, rien d'autre
    - Si l'utilisateur pose une question trop générale sur plusieurs champs sans préciser ce qu'il souhaite (comme "donne moi les details de la commande"), n'extrait rien
    - N'ajoute jamais des valeurs de champs qui ne sont pas mentionnées dans la question de l'utilisateur
    - Pas de texte avant ou après le JSON
    - Pas de commentaires dans le JSON
    - Pas de champs dupliqués
    - Si une valeur est inconnue, utilisez null
    - Ne pas inventer de nouvelles entités ou intents
    - Il peut y avoir des prepositions (de , pour , à) avant la valeur de l'entité, parametres par exemple : "la commande de #1009", "le prix de 1400" il faut savoir extraire la valeur (#1009, 1400)
    

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
        "purchase": "commande",
        "produit": "produit",
        "produits": "produit",
        "product": "produit",
        "products": "produit",
        # vendor et customer sont la même entité
        "client": "vendor",
        "clients": "vendor",
        "customer": "vendor",
        "customers": "vendor",
        "vendor": "vendor",
        "vendors": "vendor",
        "fournisseur": "vendor",
        "fournisseurs": "vendor",
        # buyer est une entité séparée
        "acheteur": "buyer",
        "acheteurs": "buyer",
        "buyer": "buyer",
        "buyers": "buyer",
        "rfq": "RFQ",
        "rfqs": "RFQ",
        "demande": "RFQ",
        "demandes": "RFQ"
    }
    return entity_mapping.get(entity.lower(), entity.lower())


# ------------------------------
# 6️⃣ Mapping complet des query_params
# ------------------------------
def map_query_params(query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mappe tous les paramètres possibles pour les uniformiser selon la liste officielle.
    """
    mapped_params = {}
    
    param_mapping = {
        # Paramètres de priorité
        "priorité": "priority",
        "urgence": "priority",
        "urgent": "priority",
        "priority": "priority",
        
        # Références
        "référence": "reference",
        "ref": "reference",
        "numero": "reference",
        "numéro": "reference",
        "number": "reference",
        "reference": "reference",
        "id": "reference",
        
        # Référence fournisseur
        "référence_fournisseur": "vendor_reference",
        "ref_fournisseur": "vendor_reference",
        "vendor_ref": "vendor_reference",
        "vendor_reference": "vendor_reference",
        
        # Fournisseur
        "fournisseur": "vendor",
        "vendor": "vendor",
        "supplier": "vendor",
        
        # Acheteur
        "acheteur": "buyer",
        "buyer": "buyer",
        "purchaser": "buyer",
        
        # Dates
        "date_limite": "order_deadline",
        "deadline": "order_deadline",
        "échéance": "order_deadline",
        "order_deadline": "order_deadline",
        "date_confirmation": "confirmation_date",
        "confirmation_date": "confirmation_date",
        "date_arrivée": "expected_arrival",
        "arrivée_prévue": "expected_arrival",
        "expected_arrival": "expected_arrival",
        "date_échéance": "due_date",
        "due_date": "due_date",
        
        # Montants
        "prix": "total",
        "price": "total",
        "montant": "total",
        "total": "total",
        "amount": "total",
        "hors_taxe": "untaxed",
        "untaxed": "untaxed",
        "ht": "untaxed",
        "tax_excluded": "tax_excluded",
        "prix_unitaire": "unitprice",
        "unitprice": "unitprice",
        "unit_price": "unitprice",
        "taxe": "taxes",
        "taxes": "taxes",
        "tax": "taxes",
        "remise": "disc",
        "discount": "disc",
        "disc": "disc",
        
        # Statuts et états
        "statut": "billing_status",
        "status": "billing_status",
        "état": "billing_status",
        "billing_status": "billing_status",
        "brouillon": "draft",
        "draft": "draft",
        "validé": "posted",
        "posted": "posted",
        "confirmé": "posted",
        
        # Produits et descriptions
        "produit": "products",
        "produits": "products",
        "products": "products",
        "product": "products",
        "description": "description",
        "desc": "description",
        "libellé": "description",
        
        # Quantités
        "quantité": "quantity",
        "quantity": "quantity",
        "qty": "quantity",
        "qte": "quantity",
        "reçu": "received",
        "received": "received",
        "livré": "received",
        "facturé": "billed",
        "billed": "billed",
        
        # Conditions et termes
        "conditions_paiement": "payment_terms",
        "payment_terms": "payment_terms",
        "termes_paiement": "payment_terms",
        "position_fiscale": "fiscal_position",
        "fiscal_position": "fiscal_position",
        "paiement": "payment",
        "payment": "payment",
        
        # Client/Customer (même chose que vendor dans ce contexte)
        "client": "vendor",
        "customer": "vendor",
        
        # Activités et documents
        "activités": "activities",
        "activities": "activities",
        "activité": "activities",
        "document_source": "source_document",
        "source_document": "source_document",
        "source": "source_document",
        
        # Métriques avancées
        "valeur_commande_moyenne": "avg_order_value",
        "avg_order_value": "avg_order_value",
        "acheté_7_jours": "purchased_last_7_days",
        "purchased_last_7_days": "purchased_last_7_days",
        "délai_achat": "lead_time_to_purchase",
        "lead_time_to_purchase": "lead_time_to_purchase",
        "rfq_7_jours": "rfqs_sent_last_7_days",
        "rfqs_sent_last_7_days": "rfqs_sent_last_7_days"
    }
    
    for key, value in query_params.items():
        # Normaliser la clé (minuscules, espaces remplacés par underscores)
        normalized_key = key.lower().replace(" ", "_").replace("-", "_")
        
        # Mapper la clé
        mapped_key = param_mapping.get(normalized_key, normalized_key)
        mapped_params[mapped_key] = value
    
    return mapped_params


# ------------------------------
# 7️⃣ Simulation backend complète
# ------------------------------
def simulated_backend_call(entity: str, intent: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simule un appel backend complet avec tous les paramètres possibles.
    """
    print(f"\n--- Appel Backend Simulé ---")
    print(f"Entité: {entity}")
    print(f"Intention: {intent}")
    print(f"Paramètres: {query_params}")
    print(f"--------------------------")

    # Mapper les paramètres d'entrée
    query_params = map_query_params(query_params)
    
    payload = {"status": 200}

    # TRAITEMENT DES COMMANDES
    if entity == "commande":
        if intent == "get":
            # Construire la réponse selon les paramètres demandés
            if query_params.get("reference"):
                payload["reference"] = query_params["reference"]
                payload["vendor"] = query_params.get("vendor", "TechCorp")
                payload["total"] = query_params.get("total", 1500.00)
                payload["billing_status"] = query_params.get("billing_status", "en_attente")
                
            if query_params.get("priority"):
                payload["priority"] = query_params["priority"]
                payload["orders_found"] = 3
                
            if query_params.get("vendor"):
                payload["vendor"] = query_params["vendor"]
                payload["orders_count"] = 5
                payload["avg_order_value"] = 2500.00
                
            # Dates et délais
            if query_params.get("order_deadline"):
                payload["order_deadline"] = query_params["order_deadline"]
            if query_params.get("confirmation_date"):
                payload["confirmation_date"] = query_params["confirmation_date"]
            if query_params.get("expected_arrival"):
                payload["expected_arrival"] = query_params["expected_arrival"]
                
        elif intent == "post":
            required_fields = ["vendor", "products", "total"]
            missing_fields = [field for field in required_fields if not query_params.get(field)]
            
            if missing_fields:
                payload["status"] = 400
                payload["message"] = f"Champs requis manquants: {', '.join(missing_fields)}"
            else:
                payload["reference"] = f"PO{hash(str(query_params)) % 10000:04d}"
                payload["vendor"] = query_params["vendor"]
                payload["total"] = query_params["total"]
                payload["billing_status"] = "draft"
                
        elif intent in ["put", "delete"]:
            if not query_params.get("reference"):
                payload["status"] = 400
                payload["message"] = "Référence de commande requise"
            else:
                action = "mise à jour" if intent == "put" else "suppression"
                payload["message"] = f"Commande {query_params['reference']} - {action} effectuée"

    # TRAITEMENT DES PRODUITS
    elif entity == "produit":
        if intent == "get":
            if query_params.get("products"):
                payload["products"] = query_params["products"]
                payload["description"] = query_params.get("description", "Description standard")
                payload["unitprice"] = query_params.get("unitprice", 299.99)
                payload["quantity"] = query_params.get("quantity", 10)
                
        elif intent == "post":
            required_fields = ["products", "unitprice"]
            missing_fields = [field for field in required_fields if not query_params.get(field)]
            
            if missing_fields:
                payload["status"] = 400
                payload["message"] = f"Champs requis manquants: {', '.join(missing_fields)}"
            else:
                payload["product_id"] = f"PROD{hash(str(query_params)) % 1000:03d}"
                payload["products"] = query_params["products"]
                payload["unitprice"] = query_params["unitprice"]

    # TRAITEMENT DES VENDORS (incluant customers)
    elif entity == "vendor":
        if intent == "get":
            if query_params.get("vendor"):
                payload["vendor"] = query_params["vendor"]
                payload["vendor_reference"] = query_params.get("vendor_reference", "V001")
                payload["lead_time_to_purchase"] = query_params.get("lead_time_to_purchase", 7)
                payload["avg_order_value"] = query_params.get("avg_order_value", 1800.00)
                payload["purchased_last_7_days"] = query_params.get("purchased_last_7_days", 2)
                payload["fiscal_position"] = query_params.get("fiscal_position", "Standard")
                payload["payment_terms"] = query_params.get("payment_terms", "30 jours")

    # TRAITEMENT DES BUYERS (acheteurs)
    elif entity == "buyer":
        if intent == "get":
            if query_params.get("buyer"):
                payload["buyer"] = query_params["buyer"]
                payload["orders_managed"] = 15
                payload["total_purchases_this_month"] = 45000.00
            else:
                payload["all_buyers"] = ["Jean Dupont", "Marie Martin", "Pierre Durand"]
                payload["active_buyers_count"] = 3

    # TRAITEMENT DES RFQ
    elif entity == "RFQ":
        if intent == "get":
            payload["rfqs_sent_last_7_days"] = query_params.get("rfqs_sent_last_7_days", 4)
            payload["vendor"] = query_params.get("vendor", "Tous fournisseurs")
            
        elif intent == "post":
            payload["rfq_id"] = f"RFQ{hash(str(query_params)) % 1000:03d}"
            payload["vendor"] = query_params.get("vendor", "À définir")
            payload["products"] = query_params.get("products", "À spécifier")

    # TRAITEMENT DES CLIENTS
    elif entity == "client":
        if intent == "get":
            if query_params.get("customer"):
                payload["customer"] = query_params["customer"]
                payload["fiscal_position"] = query_params.get("fiscal_position", "Standard")
                payload["payment_terms"] = query_params.get("payment_terms", "30 jours")

    # Entité non reconnue
    else:
        payload["status"] = 404
        payload["message"] = f"Entité '{entity}' non supportée"

    return payload


# ------------------------------
# 8️⃣ Prompt de reformulation
# ------------------------------
final_response_prompt = PromptTemplate.from_template("""
Tu es un assistant Odoo pour le module Purchase.
Voici la question : "{question}"
Voici la réponse backend : "{backend_raw}"
                                                    
Reformule cette réponse en UNE phrase simple et naturelle.
Réponds de manière claire et directe sans complexité en une seule phrase.

RÈGLES STRICTES :
- Ne t'éloigne JAMAIS de l'information fournie
- Réponds UNIQUEMENT avec UNE phrase courte et directe
- N'ajoute JAMAIS de questions de suivi
- N'ajoute JAMAIS des informations supplémentaires
- N'explique JAMAIS de détails techniques  
- N'utilise JAMAIS de termes comme "environ", "approximativement"
- Si une information est manquante, dis simplement "L'information n'est pas disponible"
- Si l'utilisateur pose une question trop générale sur plusieurs champs sans préciser ce qu'il souhaite, 
  demande-lui de clarifier en proposant les champs disponibles

RÉPONSE (une seule phrase) :                                                    
""")


# ------------------------------
# 9️⃣ Fonction principale
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

        # ➜ Résultat complet
        result = {
            "question": parsed_output['question'],
            "entity": parsed_output['entity'],
            "intent": parsed_output['intent'],
            "query_params": parsed_output['query_params'],
            "backend_response": backend_payload,
            "final_response": final_response.content
        }

        print(f"\n💬 Réponse finale : {result['final_response']}")
        return result

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return {"error": str(e)}


# ------------------------------
# 🔟 Tests étendus
# ------------------------------
if __name__ == "__main__":
    # Tests avec les nouveaux paramètres
    run_chatbot("Afficher les commandes urgentes du fournisseur TechSolutions")
    run_chatbot("Quelle est la référence fournisseur de la commande PO1234?")
    run_chatbot("Créer une commande avec le fournisseur ABC Corp pour 2500 euros")
    run_chatbot("Combien de RFQ ont été envoyées cette semaine?")
    run_chatbot("Quel est le délai d'achat moyen pour le fournisseur XYZ?")
    run_chatbot("Afficher les produits avec une quantité supérieure à 50")
    run_chatbot("Quelle est la date de confirmation de la commande REF123?")
    # Tests spécifiques buyer
    run_chatbot("Qui est l'acheteur de la commande PO456?")
    run_chatbot("Afficher toutes les commandes de l'acheteur Jean Dupont")
    run_chatbot("Quel acheteur gère le plus de commandes?")