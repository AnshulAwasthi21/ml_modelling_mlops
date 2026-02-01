# Kserve Demonstration for Intent Classifier model

This repo documents a hands-on MLOps/Kubernetes model serving demo using **KServe** on a local **KIND** Kubernetes cluster running inside **WSL Ubuntu**.

Goal: Deploy a sample **Intent classifier** as a **KServe InferenceService**, expose it locally, and run inference using `curl`.

### Create a KIND Kubernetes Cluster
```bash
kind create cluster --name=mlops-intent-model
```

### Install Cert Manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
```

### Install KServe CRDs

```bash
kubectl create namespace kserve

helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
  --version v0.16.0 \
  -n kserve \
  --wait
```

### Install KServe controller

```bash
helm install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.16.0 \
  -n kserve \
  --set kserve.controller.deploymentMode=RawDeployment \
  --wait
```

### Deploy the Intent Classifier model

```bash
kubectl create namespace intent

cat <<EOF | kubectl apply -n intent -f -
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: intent-classifier
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "$DOWNLODABLE_LOCATION"
      resources:
        requests:
          cpu: "100m"
          memory: "512Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
EOF

kubectl get inferenceservice -n intent
# this should give you the intent-classifier, inferenceservice
NAME                URL                                           READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION   AGE
intent-classifier   http://intent-classifier-intent.example.com   True                                                                  35m
```
The **$DOWNLODABLE_LOCATION** url, can you obtained from the release binary by uploading the model's pkl file in the github repo. Since, we are creating this model in our local k8s cluster, so replace the location with the github url that has the model's release binary in the same repo.

### Checkout if Kserve controller identified the InferenceService using below commands:
```bash
kubectl get pods -n kserve
NAME                                         READY   STATUS    RESTARTS   AGE
kserve-controller-manager-XXXXX              2/2     Running   0          33m
```

- Checkout the logs for the container which should give you more details to audit if it started creating deployment, service, HPA etc or not
```bash
kubectl logs kserve-controller-manager-XXXXX --all-containers -n kserve
```

### Port-forward to access the model

```bash
kubectl get all -n intent
```
This command will list all the resource under intent namespace.
From this list, grab the service name and supply its value to the kubectl port-forward command
```bash
kubectl port-forward svc/<svc-name> 8080:80 --address 0.0.0.0 -n intent
```

### Inference the Model

```bash
curl -s -X POST http://localhost:8080/v1/models/intent-classifier:predict \
  -H "Content-Type: application/json" \
  -d '{"instances":["I want to upgrade my subscription?"]}' | jq
```

<hr>

### Cleanup
```bash
# Delete the InferenceService:
kubectl delete inferenceservice intent-classifier -n intent
# Delete namespaces:
kubectl delete ns intent
kubectl delete ns kserve
kubectl delete ns cert-manager
```
Or delete the entire KIND cluster (fastest full cleanup):
```bash
kind delete cluster --name mlops-intent-model
```
