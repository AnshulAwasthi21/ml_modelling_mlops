### Model Deployment with KServe on Kubernetes

This folder contains hands-on experiments deploying machine-learning models on Kubernetes using KServe and local clusters powered by Kind / Minikube.


The goal of these labs is to simulate production-style MLOps workflows:
<ui>
- Model packaging
- InferenceService definitions
- Resource management
- Autoscaling
- Networking & port-forwarding
- Controller-driven reconciliation
- Debugging deployments
- Serving predictions via REST APIs
</ui>

These experiments complement my broader experience across DevOps, Automation, Cloud Infrastructure, and MLOps, and demonstrate how traditional platform engineering skills translate into modern ML platform operations.

<pre>
model-deployment-kserve/
│
├── sample_iris_model/
│   └── README.md        # KServe deployment of a Scikit-Learn Iris model
│
├── intent_model/
│   └── README.md        # A lite weight NLP intent classifier served via KServe
│
└── README.md            # (this file)
</pre>
