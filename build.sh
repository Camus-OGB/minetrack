#!/bin/bash

# Script de déploiement pour Render.com
echo "🚀 Starting Minetrack API deployment..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Check if models exist
if [ ! -f "app/models/best.pt" ]; then
    echo "⚠️  Warning: YOLOv8 model not found at app/models/best.pt"
fi

if [ ! -f "app/models/mlp_model.joblib" ]; then
    echo "⚠️  Warning: MLP model not found at app/models/mlp_model.joblib"
fi

echo "✅ Deployment preparation complete!"
