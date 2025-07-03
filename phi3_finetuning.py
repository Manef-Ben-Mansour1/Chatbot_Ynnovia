import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
from typing import Dict, List, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OdooChatbotTrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_model_and_tokenizer(self):
        """Load the base Phi-3 Mini model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora_config(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def create_training_prompt(self, sample: Dict[str, Any], mode: str) -> str:
        """
        Create structured prompts for training
        
        Args:
            sample: Training sample from dataset
            mode: 'intent_extraction' or 'response_formatting'
        """
        if mode == "intent_extraction":
            # Improved Intent and Query Parameter Extraction
            prompt = f"""<|system|>
Vous êtes un assistant de chatbot qui analyse les requêtes utilisateur pour identifier :
- l'intention (intent)
- l'entité principale (entity)
- les paramètres pertinents (query_params)

RÈGLES STRICTES :
- Intent possibles UNIQUEMENT : 'post', 'get', 'put', 'delete'
- Entités possibles UNIQUEMENT : 'commande', 'produit', 'vendor', 'buyer', 'RFQ'
- Paramètres autorisés UNIQUEMENT (il faut choisir à partir de cette liste tout dépend de la question de l'utilisateur) : 
  * pour une commande : total, priority, reference, vendor_reference, vendor, buyer, order_deadline, activities, source_document, untaxed, billing_status (nothing to bill or waiting to bill), confirmation_date, expected_arrival, status (cancelled or purchase order), payment_terms, payment (paid / not paid)
  * pour les informations produit d'une commande : product_name, description, quantity, received, billed, unitprice, taxes, disc, fiscal_position, due_date, tax_excluded
  * pour toutes les commandes : avg_order_value, purchased_last_7_days, lead_time_to_purchase, rfqs_sent_last_7_days

IMPORTANT :
- Générez UNIQUEMENT un JSON valide, rien d'autre
- Si l'utilisateur pose une question trop générale sur plusieurs champs sans préciser ce qu'il souhaite, n'extrait rien
- N'ajoute jamais des valeurs de champs qui ne sont pas mentionnées dans la question de l'utilisateur
- Pas de texte avant ou après le JSON
- Pas de commentaires dans le JSON
- Pas de champs dupliqués
- Si une valeur est inconnue, utilisez null
- Ne pas inventer de nouvelles entités ou intents
- Il peut y avoir des prépositions (de, pour, à) avant la valeur de l'entité, paramètres par exemple : "la commande de #1009", "le prix de 1400" il faut savoir extraire la valeur (#1009, 1400)

Retournez un JSON strictement conforme au schéma suivant :
{{
    "intent": "get|post|put|delete",
    "entity": "commande|produit|vendor|buyer|RFQ",
    "query_params": {{}}
}}
<|end|>

<|user|>
{sample['question']}
<|end|>

<|assistant|>
{{
    "intent": "{sample['intent']}",
    "entity": "{sample['entity']}",
    "query_params": {json.dumps(sample['query_params'])}
}}
<|end|>"""

        elif mode == "response_formatting":
            # Improved Response Formatting
            prompt = f"""<|system|>
Tu es un assistant Odoo pour le module Purchase.
Voici la question utilisateur et la réponse backend que tu dois reformuler.

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
- Si l'utilisateur pose une question trop générale sur plusieurs champs sans préciser ce qu'il souhaite, demande-lui de clarifier en proposant les champs disponibles
- Format des nombres approprié (devise, dates, quantités)
- Soit conversationnel mais professionnel
- Mets en évidence les informations clés pertinentes à la requête de l'utilisateur
<|end|>

<|user|>
Question: {sample['question']}

Réponse backend: {json.dumps(sample['backend_response'])}

Reformule cette réponse en une seule phrase simple et naturelle.
<|end|>

<|assistant|>
{sample['final_response']}
<|end|>"""

        return prompt
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """Load and prepare the training dataset"""
        logger.info(f"Loading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create training samples for both modes
        training_samples = []
        
        for sample in data:
            # Add intent extraction sample
            intent_prompt = self.create_training_prompt(sample, "intent_extraction")
            training_samples.append({"text": intent_prompt})
            
            # Add response formatting sample
            response_prompt = self.create_training_prompt(sample, "response_formatting")
            training_samples.append({"text": response_prompt})
        
        logger.info(f"Created {len(training_samples)} training samples")
        return Dataset.from_list(training_samples)
    
    def tokenize_function(self, examples):
        """Tokenize training examples"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train_model(self, dataset_path: str, output_dir: str = "./phi3-odoo-chatbot"):
        """Main training function"""
        logger.info("Starting model training")
        
        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.setup_lora_config()
        
        # Load and prepare dataset
        dataset = self.load_dataset(dataset_path)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Training started...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to: {output_dir}")
    
    def save_model_for_inference(self, output_dir: str, inference_dir: str = "./phi3-odoo-inference"):
        """Save the model in a format ready for inference"""
        logger.info("Preparing model for inference...")
        
        # Merge LoRA weights with base model
        merged_model = self.peft_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(inference_dir)
        self.tokenizer.save_pretrained(inference_dir)
        
        logger.info(f"Inference-ready model saved to: {inference_dir}")

class OdooInferenceEngine:
    """Class for running inference with the fine-tuned model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_intent_and_params(self, user_question: str) -> Dict[str, Any]:
        """Extract intent and query parameters from user question"""
        prompt = f"""<|system|>
Vous êtes un assistant de chatbot qui analyse les requêtes utilisateur pour identifier :
- l'intention (intent)
- l'entité principale (entity)
- les paramètres pertinents (query_params)

RÈGLES STRICTES :
- Intent possibles UNIQUEMENT : 'post', 'get', 'put', 'delete'
- Entités possibles UNIQUEMENT : 'commande', 'produit', 'vendor', 'buyer', 'RFQ'
- Paramètres autorisés UNIQUEMENT (il faut choisir à partir de cette liste tout dépend de la question de l'utilisateur) : 
  * pour une commande : total, priority, reference, vendor_reference, vendor, buyer, order_deadline, activities, source_document, untaxed, billing_status (nothing to bill or waiting to bill), confirmation_date, expected_arrival, status (cancelled or purchase order), payment_terms, payment (paid / not paid)
  * pour les informations produit d'une commande : product_name, description, quantity, received, billed, unitprice, taxes, disc, fiscal_position, due_date, tax_excluded
  * pour toutes les commandes : avg_order_value, purchased_last_7_days, lead_time_to_purchase, rfqs_sent_last_7_days

IMPORTANT :
- Générez UNIQUEMENT un JSON valide, rien d'autre
- Si l'utilisateur pose une question trop générale sur plusieurs champs sans préciser ce qu'il souhaite, n'extrait rien
- N'ajoute jamais des valeurs de champs qui ne sont pas mentionnées dans la question de l'utilisateur
- Pas de texte avant ou après le JSON
- Pas de commentaires dans le JSON
- Pas de champs dupliqués
- Si une valeur est inconnue, utilisez null
- Ne pas inventer de nouvelles entités ou intents
- Il peut y avoir des prépositions (de, pour, à) avant la valeur de l'entité, paramètres par exemple : "la commande de #1009", "le prix de 1400" il faut savoir extraire la valeur (#1009, 1400)

Retournez un JSON strictement conforme au schéma suivant :
{{
    "intent": "get|post|put|delete",
    "entity": "commande|produit|vendor|buyer|RFQ",
    "query_params": {{}}
}}
<|end|>

<|user|>
{user_question}
<|end|>

<|assistant|>
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except:
            logger.error(f"Failed to parse JSON from response: {response}")
            return {"intent": "get", "entity": "commande", "query_params": {}}
    
    def format_response(self, user_question: str, backend_data: Dict[str, Any]) -> str:
        """Format backend response into natural language"""
        prompt = f"""<|system|>
Tu es un assistant Odoo pour le module Purchase.
Voici la question utilisateur et la réponse backend que tu dois reformuler.

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
- Si l'utilisateur pose une question trop générale sur plusieurs champs sans préciser ce qu'il souhaite, demande-lui de clarifier en proposant les champs disponibles
- Format des nombres approprié (devise, dates, quantités)
- Soit conversationnel mais professionnel
- Mets en évidence les informations clés pertinentes à la requête de l'utilisateur
<|end|>

<|user|>
Question: {user_question}

Réponse backend: {json.dumps(backend_data)}

Reformule cette réponse en une seule phrase simple et naturelle.
<|end|>

<|assistant|>
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

def main():
    """Main function to run the training"""
    # Initialize trainer
    trainer = OdooChatbotTrainer()
    
    # Train the model
    dataset_path = "odoo_data.json"  # Path to your dataset
    output_dir = "./phi3-odoo-chatbot"
    inference_dir = "./phi3-odoo-inference"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        logger.info("Please ensure your odoo_data.json file is in the current directory")
        return
    
    try:
        # Train the model
        trainer.train_model(dataset_path, output_dir)
        
        # Prepare model for inference
        trainer.save_model_for_inference(output_dir, inference_dir)
        
        logger.info("Training and preparation completed successfully!")
        
        # Example of how to use the trained model
        logger.info("Testing the trained model...")
        inference_engine = OdooInferenceEngine(inference_dir)
        
        # Test intent extraction
        test_question = "Afficher les commandes urgentes du fournisseur TechSolutions"
        extracted = inference_engine.extract_intent_and_params(test_question)
        logger.info(f"Extracted: {extracted}")
        
        # Test response formatting
        sample_backend_data = {
            "orders": [
                {"reference": "PO001", "vendor": "TechSolutions", "total": 1500.00, "status": "purchase order", "priority": "urgent"}
            ]
        }
        formatted_response = inference_engine.format_response(test_question, sample_backend_data)
        logger.info(f"Formatted response: {formatted_response}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()