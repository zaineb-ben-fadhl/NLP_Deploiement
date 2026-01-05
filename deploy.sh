#!/usr/bin/env bash
set -euo pipefail

#################################
# VARIABLES
#################################
RESOURCE_GROUP="rg-dreaddit-mlops"
LOCATION="francecentral"  # RG existant ou nouvelle création
ACR_NAME="acrdreaddit54381" # Nom fixe pour réutilisation
CONTAINER_APP_NAME="dreaddit-api"
CONTAINERAPPS_ENV="env-dreaddit-mlops"
IMAGE_NAME="dreaddit-apivf"
IMAGE_TAG="v2"
TARGET_PORT=8000

LAW_REGIONS_CANDIDATES=("francecentral")
LAW_LOCATION=""

#################################
# 0) Vérification Azure CLI
#################################
echo "🔍 Vérification du contexte Azure..."
az account show --query "{name:name, cloudName:cloudName}" -o json >/dev/null
echo "✅ Azure CLI OK"

#################################
# 1) Providers
#################################
echo "🔧 Enregistrement des providers..."
az provider register --namespace Microsoft.ContainerRegistry --wait >/dev/null
az provider register --namespace Microsoft.App --wait >/dev/null
az provider register --namespace Microsoft.Web --wait >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait >/dev/null
echo "✅ Providers OK"

#################################
# 2) Resource Group
#################################
echo "📁 Vérification du groupe de ressources..."
RG_LOC=$(az group show -n "$RESOURCE_GROUP" --query location -o tsv 2>/dev/null || echo "")
if [ -n "$RG_LOC" ]; then
    LOCATION="$RG_LOC"
    echo "✅ RG existant: $RESOURCE_GROUP (location=$LOCATION)"
else
    az group create -n "$RESOURCE_GROUP" -l "$LOCATION" >/dev/null
    echo "✅ RG créé: $RESOURCE_GROUP (location=$LOCATION)"
fi

#################################
# 3) Création / Vérification ACR
#################################
echo "📦 Vérification du Container Registry: $ACR_NAME"
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
    echo "ℹ️ ACR non trouvé. Création..."
    az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$ACR_NAME" \
        --sku Basic \
        --admin-enabled true \
        --location "$LOCATION"
    echo "✅ ACR créé: $ACR_NAME"
else
    echo "✅ ACR existant: $ACR_NAME"
fi

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')
IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

#################################
# 4) Docker Build + Push
#################################
echo "🐳 Docker login vers ACR..."
docker login "$ACR_LOGIN_SERVER" -u "$ACR_USER" -p "$ACR_PASS" >/dev/null

echo "🏗️ Build Docker..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

echo "🏷️ Tag Docker..."
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$IMAGE"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"

echo "📤 Push Docker..."
docker push "$IMAGE"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
echo "✅ Image pushée : $IMAGE"

#################################
# 5) Log Analytics
#################################
LAW_NAME="law-dreaddit"
for r in "${LAW_REGIONS_CANDIDATES[@]}"; do
    if ! az monitor log-analytics workspace show --resource-group "$RESOURCE_GROUP" --workspace-name "$LAW_NAME" >/dev/null 2>&1; then
        az monitor log-analytics workspace create \
            --resource-group "$RESOURCE_GROUP" \
            --workspace-name "$LAW_NAME" \
            --location "$r"
        LAW_LOCATION="$r"
        echo "✅ Log Analytics créé : $LAW_NAME (region=$LAW_LOCATION)"
        break
    else
        LAW_LOCATION="$r"
        echo "✅ Log Analytics existant : $LAW_NAME"
    fi
done

LAW_ID=$(az monitor log-analytics workspace show --resource-group "$RESOURCE_GROUP" --workspace-name "$LAW_NAME" --query customerId -o tsv | tr -d '\r')
LAW_KEY=$(az monitor log-analytics workspace get-shared-keys --resource-group "$RESOURCE_GROUP" --workspace-name "$LAW_NAME" --query primarySharedKey -o tsv | tr -d '\r')

#################################
# 6) Container Apps Environment
#################################
SUB_ID=$(az account show --query id -o tsv | tr -d '\r')
ENV_ID="/subscriptions/$SUB_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/managedEnvironments/$CONTAINERAPPS_ENV"
APP_ID="/subscriptions/$SUB_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.App/containerApps/$CONTAINER_APP_NAME"

echo "🌿 Création / validation Container Apps Environment..."
az rest --method put \
  --headers "Content-Type=application/json" \
  --url "https://management.azure.com$ENV_ID?api-version=2023-05-01" \
  --body "{
      \"location\": \"$LOCATION\",
      \"properties\": {
          \"appLogsConfiguration\": {
              \"destination\": \"log-analytics\",
              \"logAnalyticsConfiguration\": {
                  \"customerId\": \"$LAW_ID\",
                  \"sharedKey\": \"$LAW_KEY\"
              }
          }
      }
  }"

#################################
# 7) Déploiement Container App
#################################
echo "🚀 Déploiement Container App..."
az rest --method put \
  --headers "Content-Type=application/json" \
  --url "https://management.azure.com$APP_ID?api-version=2023-05-01" \
  --body "{
      \"location\": \"$LOCATION\",
      \"properties\": {
          \"managedEnvironmentId\": \"$ENV_ID\",
          \"configuration\": {
              \"ingress\": {\"external\": true, \"targetPort\": $TARGET_PORT, \"transport\": \"auto\"},
              \"secrets\": [{\"name\": \"acr-pwd\", \"value\": \"$ACR_PASS\"}],
              \"registries\": [{\"server\": \"$ACR_LOGIN_SERVER\", \"username\": \"$ACR_USER\", \"passwordSecretRef\": \"acr-pwd\"}]
          },
          \"template\": {
              \"containers\": [{\"name\": \"api\", \"image\": \"$IMAGE\", \"resources\": {\"cpu\": 0.5, \"memory\": \"1Gi\"}}],
              \"scale\": {\"minReplicas\": 1, \"maxReplicas\": 1}
          }
      }
  }"

#################################
# 8) URL API
#################################
APP_URL=$(az rest --method get --url "https://management.azure.com$APP_ID?api-version=2023-05-01" --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')

echo ""
echo "=========================================="
echo "✅ DÉPLOIEMENT RÉUSSI"
echo "=========================================="
echo "RG Region      : $LOCATION"
echo "ACR            : $ACR_NAME"
echo "Image          : $IMAGE"
echo ""
echo "URLs :"
echo "  API      : https://$APP_URL"
echo "  Health   : https://$APP_URL/health"
echo "  Docs     : https://$APP_URL/docs"
echo ""
echo "Pour supprimer toutes les ressources :"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo "=========================================="
