#!/bin/bash
# Generate kubectl outputs for assignment submission

echo "# Kubectl Outputs" > kubectl_outputs.md
echo "" >> kubectl_outputs.md
echo "Generated on: $(date)" >> kubectl_outputs.md
echo "" >> kubectl_outputs.md

echo "## Deployment Description" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
sg docker -c "kubectl describe deployment dog-classifier-deployment" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
echo "" >> kubectl_outputs.md

echo "## Pod Description" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
POD_NAME=$(sg docker -c "kubectl get pods -l app=dog-classifier -o jsonpath='{.items[0].metadata.name}'")
sg docker -c "kubectl describe pod $POD_NAME" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
echo "" >> kubectl_outputs.md

echo "## Ingress Description" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
sg docker -c "kubectl describe ingress dog-classifier-ingress" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
echo "" >> kubectl_outputs.md

echo "## Pod Metrics (kubectl top pod)" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
sg docker -c "kubectl top pod" >> kubectl_outputs.md 2>&1
echo '```' >> kubectl_outputs.md
echo "" >> kubectl_outputs.md

echo "## Node Metrics (kubectl top node)" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md
sg docker -c "kubectl top node" >> kubectl_outputs.md 2>&1
echo '```' >> kubectl_outputs.md
echo "" >> kubectl_outputs.md

echo "## All Resources (kubectl get all -o yaml)" >> kubectl_outputs.md
echo '```yaml' >> kubectl_outputs.md
sg docker -c "kubectl get all -o yaml" >> kubectl_outputs.md
echo '```' >> kubectl_outputs.md

echo "Kubectl outputs generated successfully!"
