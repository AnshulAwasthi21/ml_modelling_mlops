# MLOps | Kubernetes Model Deployment Lab

Hands-on projects exploring the full ML lifecycle — from model training and versioning to containerized deployment and Kubernetes-native inference using **KServe**.

This repository reflects my work across **In IT & Automation**, combined with modern **DevOps, Cloud, and MLOps practices**, where I routinely build production-style ML platforms and internal POCs.

---

## Repository Structure

<pre>
ml_modelling_mlops/
├── iris_model/
├── wine_prediction/
├── intent-classifier-model-k8s/
├── model-deployment-kserve/
│ ├── sample_iris_model/
│ └── intent_model/
├── .dvc/
├── .gitignore
└── README.md
</pre>


---

## What This Covers

- ML model training & evaluation
- Dataset / artifact versioning with DVC
- Flask-based inference APIs
- Dockerized services
- Kubernetes deployments
- KServe `InferenceService`
- Autoscaling & resource tuning
- Debugging real-world failures
- Local clusters via Kind / Minikube

---

## Tech Stack

**ML:** Scikit-Learn, Pandas  
**Tracking:** DVC  
**Serving:** Flask, REST  
**Orchestration:** Docker, Kubernetes, Helm  
**Model Serving:** KServe, cert-manager  
**Local Infra:** Kind / Minikube, WSL

---

## Getting Started

Each folder contains a dedicated README with:

- setup steps
- deployment workflow
- inference testing
- cleanup commands

Start here:

```bash
cd model-deployment-kserve/sample_iris_model
```

Roadmap:-
<ui>
- GPU inference services
- EKS deployments
- Canary rollouts
- CI/CD pipelines
- Monitoring & drift detection
</ui>



