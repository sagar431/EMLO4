echo "# Kubernetes Deployment Status" > deployment_status.md
echo "" >> deployment_status.md

echo "## Deployment Description" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
sg docker -c "kubectl describe deployment classifier-deployment" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
echo "" >> deployment_status.md

echo "## Pod Description" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
POD_NAME=$(sg docker -c "kubectl get pods -l app=classifier -o jsonpath='{.items[0].metadata.name}'")
sg docker -c "kubectl describe pod $POD_NAME" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
echo "" >> deployment_status.md

echo "## Ingress Description" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
sg docker -c "kubectl describe ingress classifier-ingress" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
echo "" >> deployment_status.md

echo "## Pod Metrics" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
sg docker -c "kubectl top pod" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
echo "" >> deployment_status.md

echo "## Node Metrics" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
sg docker -c "kubectl top node" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
echo "" >> deployment_status.md

echo "## All Resources (YAML)" >> deployment_status.md
echo "\`\`\`yaml" >> deployment_status.md
sg docker -c "kubectl get all -o yaml" >> deployment_status.md
echo "\`\`\`" >> deployment_status.md
