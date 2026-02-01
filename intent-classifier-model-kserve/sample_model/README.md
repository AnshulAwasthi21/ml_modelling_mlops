# KServe on KIND (Ubuntu) - Deploying a Sample sklearn-iris Model with InferenceService

This repo documents a hands-on MLOps/Kubernetes model serving demo using **KServe** on a local **KIND** Kubernetes cluster running inside **WSL Ubuntu**.

Goal: Deploy a sample **sklearn iris classifier** as a **KServe InferenceService**, expose it locally, and run inference using `curl`.

---

## Architecture (High-Level)

1. **KIND** creates a local Kubernetes cluster using Docker containers.
2. **cert-manager** provides certificates needed for KServe webhooks.
3. **KServe** installs:
   - CRDs (new Kubernetes APIs like `InferenceService`)
   - Controller (the brain that watches InferenceService and creates pods/services/HPAs)
4. A **KServe InferenceService** is created which:
   - Pulls the model from `storageUri`
   - Runs it using `kserve/sklearnserver`
   - Creates a Kubernetes `Deployment + Service + HPA`
5. We **port-forward** the predictor service locally and test it with `curl`.

---

## Prerequisites

- WSL Ubuntu
- Docker installed and running
- `kubectl` installed
- `helm` installed
- `kind` installed

Verify:
```bash
docker version
kubectl version --client
helm version
kind version
```

Step 1 — Create a KIND Kubernetes Cluster

kind create cluster --name=mlops-lab-kserve


Check current context:
```bash
kubectl config current-context
# kind-mlops-lab-kserve
Check nodes:
kubectl get nodes
```

At this point, a Kubernetes control-plane node is running as a Docker container.

Step 2 — Install cert-manager (Required for KServe webhooks)
```bash
cert-manager is used by KServe for webhook TLS certificates.

kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
```

Verify cert-manager pods:
```bash
kubectl get pods -n cert-manager
```

Expected:
```bash
cert-manager
cert-manager-cainjector
cert-manager-webhook
(all Running)
```
Step 3 — Create KServe Namespace
```bash
kubectl create namespace kserve
```
Step 4 — Update Helm Repos
```bash
helm repo update
```
Step 5 — Install KServe CRDs

CRDs = Custom Resource Definitions.
This step adds new Kubernetes resource types like:
- InferenceService
- ServingRuntime
- etc.

```bash
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
  --version v0.16.0 \
  -n kserve \
  --wait
```
Step 6 — Install KServe Controller

The controller is the component that:

watches InferenceService

creates Deployments/Services/HPAs/Ingress

reconciles desired vs actual state
```bash
helm install kserve oci://ghcr.io/kserve/charts/kserve \
  --version v0.16.0 \
  -n kserve \
  --set kserve.controller.deploymentMode=RawDeployment \
  --wait
```

Verify controller pod:
```bash
kubectl get pods -n kserve
```
Step 7 — Create a Namespace for ML Workloads
```bash
kubectl create namespace ml
```
Step 8 — Deploy Model using KServe InferenceService
Why this command style?
```bash
cat <<EOF | kubectl apply -n ml -f -
...
EOF
```

This is a Linux shell trick called Heredoc:

- cat <<EOF starts a multi-line input block
- Everything until EOF is treated as text output
- | pipes that text into the next command

```bash
kubectl apply -f - means:

-f = input is a file

- = file is STDIN (coming from the pipe)
```

⭐ Benefit:

You can apply YAML without creating a separate .yaml file on disk.
```bash
InferenceService YAML (as used)
cat <<EOF | kubectl apply -n ml -f -
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
      resources:
        requests:
          cpu: "100m"
          memory: "512Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
EOF
```
<pre>
Understanding the InferenceService YAML (Simple Explanation)

1) apiVersion + kind
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService

This says: create a KServe model-serving resource, not a normal Kubernetes Deployment.

2) Metadata Name
metadata:
  name: sklearn-iris

This becomes:
- service name → sklearn-iris-predictor
- deployment name → sklearn-iris-predictor

3) Predictor Block
spec:
  predictor:
    model:

Meaning: "This is the prediction container that will serve inference."

4) Model Format
modelFormat:
  name: sklearn

This tells KServe to use a compatible runtime for scikit-learn models.

5) storageUri
storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"


This is the model location. KServe will pull the model and mount it inside the serving container.

6) Resources (Important!)
resources:
  requests:
    cpu: "100m"
    memory: "512Mi"
  limits:
    cpu: "1"
    memory: "1Gi"


requests = minimum guaranteed resources for scheduling
- 100m CPU = 0.1 CPU
- 512Mi memory
- limits = max the container can use up to 1 CPU and 1Gi RAM

⭐ This protects the cluster and prevents a pod from taking unlimited resources.
</pre>

Step 9 — Verify Resources Created by KServe
Check service:
```bash
kubectl get svc -n ml

Check everything:

kubectl get all -n ml
```
You should see:
- Pod
- Service (ClusterIP)
- Deployment
- ReplicaSet
- HPA

Step 10 — Expose the Service Locally using Port Forward

Since this is a local KIND cluster, the service is only inside the cluster network.
To access it from your laptop terminal:
```bash
kubectl -n ml port-forward svc/sklearn-iris-predictor 8081:80 --address 0.0.0.0
```
This maps:
- localhost:8081 → service port 80 → container port 8080

Step 11 — Run Inference Tests

Example request:
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"instances":[[1,1,1,1]]}' \
  http://localhost:8081/v1/models/sklearn-iris:predict

Sample output:
{"predictions":[0]}
```

Another test:
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"instances":[[5,5,5,5]]}' \
  http://localhost:8081/v1/models/sklearn-iris:predict
```

<h4>Notes / Learnings</h4>

KIND is great for fast local K8s testing.
- KServe abstracts away manual Deployment/Service creation.
- InferenceService automatically creates:
- Deployment
- Service
- HPA
- (optional) Ingress

Port-forwarding is the quickest way to test locally without exposing via NodePort/LoadBalancer.

<hr>
<h4>Cleanup (Free CPU/RAM on Laptop)</h4>

```bash
# Delete the InferenceService:
kubectl delete inferenceservice sklearn-iris -n ml
# Delete namespaces:
kubectl delete ns ml
kubectl delete ns kserve
kubectl delete ns cert-manager
```
Or delete the entire KIND cluster (fastest full cleanup):
```bash
kind delete cluster --name mlops-lab-kserve
```

Note:-
**kubectl get all -n ml** generally expands to something like:
<ui>
- Pods
- Services
- Deployments
- ReplicaSets
- StatefulSets (sometimes)
- DaemonSets (sometimes)
- Jobs/CronJobs (sometimes)
- HPAs (depending on kubectl version/plugins)<br>
But it does not automatically include CRDs, like:
- inferenceservices.serving.kserve.io
- certificates.cert-manager.io
- clusterissuers.cert-manager.io
- etc.
</ui>

kubectl get all shows you the “generated children”, not the “parent CRD”.

```bash
To see all resource kinds in a namespace (including CRDs)

If you want a "true all", you can do:

kubectl api-resources --verbs=list --namespaced -o name \
| xargs -n 1 kubectl get -n ml --ignore-not-found
```

That prints everything it can list in ml, including CRDs (KServe, cert-manager, etc.).

This removes the cluster completely (all namespaces/resources inside it).
