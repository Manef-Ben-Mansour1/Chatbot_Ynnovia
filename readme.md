# Chatbot AI avec Ollama et Langchain

Ce projet utilise le modèle **phi3:mini** d'Ollama via Langchain pour construire un chatbot capable de traiter des requêtes utilisateur et de simuler des appels backend pour des entités comme **commande**, **produit**, et **client**. Le chatbot extrait des informations pertinentes des requêtes, effectue une simulation de traitement backend, et génère des réponses reformulées et compréhensibles pour l'utilisateur.

## Description

Le chatbot fonctionne en plusieurs étapes :
1. **Extraction** des entités et des paramètres d'une question utilisateur via un prompt défini.
2. **Simulation** d'un appel backend en fonction des entités et des paramètres extraits.
3. **Reformulation** de la réponse en un format naturel et clair.

Le système prend en charge plusieurs types d'intentions : **post**, **get**, **update**, et **delete**. Les entités prises en charge incluent : **commande**, **produit**, et **client**.

## Prérequis

Avant de commencer, vous devez avoir installé les outils suivants :

- **Python 3.10** 
- **pip** pour installer les dépendances Python
- Un environnement virtuel pour isoler les dépendances (optionnel mais recommandé)

## Installation

1. **Clonez le projet :**

    ```bash
    git https://github.com/Manef-Ben-Mansour1/Chatbot_Ynnovia
    ```



2. **Installez les dépendances :**

    ```bash
    pip install -r requirements.txt
    ```

    Le fichier `requirements.txt` contient les bibliothèques nécessaires :
    ```txt
    langchain
    pydantic
    langchain_ollama
    ```

3. **Configurez Ollama** :
    - Assurez-vous que vous avez accès au modèle **phi3:mini** sur Ollama. Vous pouvez consulter [Ollama](https://ollama.com) pour plus de détails sur l'accès au modèle.
    
    - Si vous ne l'avez pas encore, suivez les instructions sur leur site pour configurer l'API Ollama et obtenir un modèle.

## Utilisation

### Exécution du chatbot

Pour tester le chatbot, vous pouvez exécuter le script Python avec différentes requêtes utilisateurs. Exemple :

1. **Exécution du script :**

    ```bash
    python chatbot.py
    ```

    Le script attendra des entrées dans la console avec des questions à traiter. Par exemple :

    ```bash
    Créer une commande pour le produit X avec le statut 'en attente' pour la commande #5299 et un prix de 14950.
    ```

2. **Sortie attendue :**

    Une fois que le chatbot a traité la question, il fournira une réponse formatée et simulera un appel backend pour générer les informations appropriées.

    Exemple de sortie :
    ```text
    ❓ Question utilisateur : Créer une commande pour le produit X avec le statut 'en attente' pour la commande #5299 et un prix de 14950.
    ✅ Extraction : {'question': "Créer une commande pour le produit X...", 'entity': 'commande', 'intent': 'post', 'query_params': {'id': None, 'status': 'en attente', 'numero': '#5299', 'prix': 14950}}
    --- Appel Backend Simulé ---
    Entité: commande
    Intention: post
    Paramètres: {'id': None, 'status': 'en attente', 'numero': '#5299', 'prix': 14950}
    --------------------------
    ✅ Payload backend : {'status': 200, 'commande_id': None, 'status_commande': 'en attente', 'prix': 14950, 'produit': 'non spécifié', 'currency': 'inconnu'}
    💬 Réponse finale : Votre commande pour le produit X avec un prix fixé à 14950 est mise en attente, et bienvenue au suivi.
    ```

### Fonctionnement des étapes

1. **Extraction des entités et paramètres** : Le chatbot analyse la question pour extraire des entités comme "commande", "produit", ou "client" et enregistre les paramètres associés (comme le prix ou le numéro de commande).
   
2. **Simulation de l'appel backend** : En fonction des entités et des paramètres extraits, une simulation de backend est effectuée. Ce backend pourrait interagir avec une base de données ou un système externe dans une application réelle.

3. **Reformulation de la réponse** : La réponse brute obtenue du backend est ensuite reformulée de manière claire et naturelle pour l'utilisateur.

## Structure du Code

Voici un aperçu de la structure du code :

- `chatbot.py` : Le fichier principal contenant le code du chatbot.
- `requirements.txt` : Le fichier de dépendances.
- `README.md` : Ce fichier contenant la documentation.

