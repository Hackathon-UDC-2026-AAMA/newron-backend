#!/bin/bash

export HOST_IP=$(ipconfig getifaddr en0)

if [ -z "$HOST_IP" ]; then
    echo "❌ Error: No se pudo obtener la IP de en0. ¿Estás conectado al Wi-Fi?"
    exit 1
fi

echo "🚀 Iniciando con IP: $HOST_IP"

# --build: Reconstruye las imágenes antes de iniciar
# --remove-orphans: Limpia contenedores antiguos que ya no están en el archivo .yml
docker-compose up --build --remove-orphans