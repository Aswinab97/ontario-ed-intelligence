#!/bin/bash
# ── AZURE CONTAINER APPS DEPLOYMENT SCRIPT ──────────────────────────────────
# This script deploys the Ontario Healthcare Intelligence Platform Streamlit app
# to Azure Container Apps. 
#
# PREREQUISITES:
# 1. Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
# 2. Authenticate: `az login`

# Variables (You can customize these)
RESOURCE_GROUP="ontario-health-rg"
LOCATION="canadacentral"
ENVIRONMENT="ontario-health-env"
APP_NAME="ontario-ed-intelligence-app"
REGISTRY_NAME="ontariohealthacr$RANDOM"

echo "🏥 Starting deployment of Ontario Healthcare Intelligence Platform to Azure..."

# 1. Create Resource Group
echo "📦 Creating Resource Group: $RESOURCE_GROUP..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# 2. Create Azure Container Registry (ACR)
echo "🐳 Creating Azure Container Registry: $REGISTRY_NAME..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $REGISTRY_NAME \
    --sku Basic \
    --admin-enabled true

# 3. Build and push the Docker image to ACR using ACR Tasks
echo "🔨 Building and pushing Docker image to ACR..."
az acr build \
    --registry $REGISTRY_NAME \
    --image $APP_NAME:latest \
    --file Dockerfile.dashboard \
    .

# 4. Create Azure Container Apps Environment
echo "☁️ Creating Container Apps Environment: $ENVIRONMENT..."
az containerapp env create \
    --name $ENVIRONMENT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# 5. Deploy the Container App
echo "🚀 Deploying the Container App: $APP_NAME..."
# Retrieve ACR credentials
ACR_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query "passwords[0].value" -o tsv)

az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $ENVIRONMENT \
    --image "$REGISTRY_NAME.azurecr.io/$APP_NAME:latest" \
    --target-port 8501 \
    --ingress 'external' \
    --registry-server "$REGISTRY_NAME.azurecr.io" \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --min-replicas 1 \
    --max-replicas 5 \
    --cpu 1.0 --memory 2.0Gi

# 6. Get the FQDN (URL) of the deployed app
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)

echo "✅ Deployment Complete!"
echo "🌐 Your app is live at: https://$APP_URL"
