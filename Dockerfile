# Utiliser une image Python officielle comme image de base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de requirements.txt dans le répertoire de travail
COPY requirements.txt requirements.txt

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies if any
RUN apt-get update && apt-get install -y curl

# Install Ollama using the install script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copier le contenu de votre projet dans le répertoire de travail
COPY . .

# Copy the entrypoint script into the container
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Expose the port the app runs on
EXPOSE 5000

# Use the entrypoint script to start Ollama and the Flask app
ENTRYPOINT ["/entrypoint.sh"]