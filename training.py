import os
import torch
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def train_model_with_pdfs(pdf_folder, model_name="facebook/llama-7b", output_dir="./results"):
    # Fonction pour extraire le texte des fichiers PDF
    def extract_text_from_pdf(file_path):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    # Extraction du texte de tous les fichiers PDF
    documents = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            documents.append(extract_text_from_pdf(file_path))

    # Tokenisation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_texts = [tokenizer(doc, return_tensors='pt', padding=True, truncation=True) for doc in documents]

    # Charger le modèle pré-entraîné
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Définir un dataset personnalisé
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts

        def __len__(self):
            return len(self.tokenized_texts)

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.tokenized_texts[idx].items()}

    dataset = CustomDataset(tokenized_texts)

    # Création de l'objet Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    # Entraînement du modèle
    trainer.train()

    # Sauvegarde du modèle et du tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Appeler la fonction avec le dossier contenant les PDF
train_model_with_pdfs("pdf")
