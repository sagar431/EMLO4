# Kubernetes Deployment Status

## Deployment Description
```
Name:                   classifier-deployment
Namespace:              default
CreationTimestamp:      Thu, 15 Jan 2026 18:40:11 +0000
Labels:                 app=classifier
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               app=classifier
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
Pod Template:
  Labels:  app=classifier
  Containers:
   classifier:
    Image:         classifier-k8s:latest
    Port:          7860/TCP
    Host Port:     0/TCP
    Environment:   <none>
    Mounts:        <none>
  Volumes:         <none>
  Node-Selectors:  <none>
  Tolerations:     <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Available      True    MinimumReplicasAvailable
  Progressing    True    NewReplicaSetAvailable
OldReplicaSets:  <none>
NewReplicaSet:   classifier-deployment-68964c45b4 (2/2 replicas created)
Events:
  Type    Reason             Age    From                   Message
  ----    ------             ----   ----                   -------
  Normal  ScalingReplicaSet  2m14s  deployment-controller  Scaled up replica set classifier-deployment-68964c45b4 from 0 to 2
```

## Pod Description
```
Name:             classifier-deployment-68964c45b4-bmjz6
Namespace:        default
Priority:         0
Service Account:  default
Node:             minikube/192.168.49.2
Start Time:       Thu, 15 Jan 2026 18:40:11 +0000
Labels:           app=classifier
                  pod-template-hash=68964c45b4
Annotations:      <none>
Status:           Running
IP:               10.244.0.7
IPs:
  IP:           10.244.0.7
Controlled By:  ReplicaSet/classifier-deployment-68964c45b4
Containers:
  classifier:
    Container ID:   docker://9931ae309f7516173eded45ee1507c91ec171ce75132443217ec5104e404907d
    Image:          classifier-k8s:latest
    Image ID:       docker://sha256:afbc229215b496350962c3dd646714031ea52ad2b8aaa34b79317f1044786cf8
    Port:           7860/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Thu, 15 Jan 2026 18:40:12 +0000
    Ready:          True
    Restart Count:  0
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-2lqwq (ro)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True 
  Initialized                 True 
  Ready                       True 
  ContainersReady             True 
  PodScheduled                True 
Volumes:
  kube-api-access-2lqwq:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   BestEffort
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason     Age    From               Message
  ----    ------     ----   ----               -------
  Normal  Scheduled  2m14s  default-scheduler  Successfully assigned default/classifier-deployment-68964c45b4-bmjz6 to minikube
  Normal  Pulled     2m13s  kubelet            spec.containers{classifier}: Container image "classifier-k8s:latest" already present on machine
  Normal  Created    2m13s  kubelet            spec.containers{classifier}: Created container: classifier
  Normal  Started    2m13s  kubelet            spec.containers{classifier}: Started container classifier
```

## Ingress Description
```
Name:             classifier-ingress
Labels:           <none>
Namespace:        default
Address:          192.168.49.2
Ingress Class:    nginx
Default backend:  <default>
Rules:
  Host                  Path  Backends
  ----                  ----  --------
  classifier.localhost  
                        /   classifier-service:80 (10.244.0.7:7860,10.244.0.6:7860)
Annotations:            nginx.ingress.kubernetes.io/affinity: cookie
                        nginx.ingress.kubernetes.io/affinity-mode: balanced
                        nginx.ingress.kubernetes.io/session-cookie-expires: 172800
                        nginx.ingress.kubernetes.io/session-cookie-max-age: 172800
                        nginx.ingress.kubernetes.io/session-cookie-name: INGRESSCOOKIE
Events:
  Type    Reason  Age               From                      Message
  ----    ------  ----              ----                      -------
  Normal  Sync    77s (x2 over 2m)  nginx-ingress-controller  Scheduled for sync
```

## Pod Metrics
```
NAME                                     CPU(cores)   MEMORY(bytes)   
classifier-deployment-68964c45b4-bmjz6   35m          1211Mi          
classifier-deployment-68964c45b4-zg4rr   35m          1186Mi          
```

## Node Metrics
```
NAME       CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)   
minikube   239m         0%       2925Mi          1%          
```

## All Resources (YAML)
```yaml
apiVersion: v1
items:
- apiVersion: v1
  kind: Pod
  metadata:
    creationTimestamp: "2026-01-15T18:40:11Z"
    generateName: classifier-deployment-68964c45b4-
    generation: 1
    labels:
      app: classifier
      pod-template-hash: 68964c45b4
    name: classifier-deployment-68964c45b4-bmjz6
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: ReplicaSet
      name: classifier-deployment-68964c45b4
      uid: 1d9d56f6-6c95-4580-a01f-993141a2abec
    resourceVersion: "1344"
    uid: c2cade03-c695-433e-beed-aaefd9d223cf
  spec:
    containers:
    - image: classifier-k8s:latest
      imagePullPolicy: Never
      name: classifier
      ports:
      - containerPort: 7860
        protocol: TCP
      resources: {}
      terminationMessagePath: /dev/termination-log
      terminationMessagePolicy: File
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-2lqwq
        readOnly: true
    dnsPolicy: ClusterFirst
    enableServiceLinks: true
    nodeName: minikube
    preemptionPolicy: PreemptLowerPriority
    priority: 0
    restartPolicy: Always
    schedulerName: default-scheduler
    securityContext: {}
    serviceAccount: default
    serviceAccountName: default
    terminationGracePeriodSeconds: 30
    tolerations:
    - effect: NoExecute
      key: node.kubernetes.io/not-ready
      operator: Exists
      tolerationSeconds: 300
    - effect: NoExecute
      key: node.kubernetes.io/unreachable
      operator: Exists
      tolerationSeconds: 300
    volumes:
    - name: kube-api-access-2lqwq
      projected:
        defaultMode: 420
        sources:
        - serviceAccountToken:
            expirationSeconds: 3607
            path: token
        - configMap:
            items:
            - key: ca.crt
              path: ca.crt
            name: kube-root-ca.crt
        - downwardAPI:
            items:
            - fieldRef:
                apiVersion: v1
                fieldPath: metadata.namespace
              path: namespace
  status:
    conditions:
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:13Z"
      observedGeneration: 1
      status: "True"
      type: PodReadyToStartContainers
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:11Z"
      observedGeneration: 1
      status: "True"
      type: Initialized
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:13Z"
      observedGeneration: 1
      status: "True"
      type: Ready
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:13Z"
      observedGeneration: 1
      status: "True"
      type: ContainersReady
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:11Z"
      observedGeneration: 1
      status: "True"
      type: PodScheduled
    containerStatuses:
    - containerID: docker://9931ae309f7516173eded45ee1507c91ec171ce75132443217ec5104e404907d
      image: classifier-k8s:latest
      imageID: docker://sha256:afbc229215b496350962c3dd646714031ea52ad2b8aaa34b79317f1044786cf8
      lastState: {}
      name: classifier
      ready: true
      resources: {}
      restartCount: 0
      started: true
      state:
        running:
          startedAt: "2026-01-15T18:40:12Z"
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-2lqwq
        readOnly: true
        recursiveReadOnly: Disabled
    hostIP: 192.168.49.2
    hostIPs:
    - ip: 192.168.49.2
    observedGeneration: 1
    phase: Running
    podIP: 10.244.0.7
    podIPs:
    - ip: 10.244.0.7
    qosClass: BestEffort
    startTime: "2026-01-15T18:40:11Z"
- apiVersion: v1
  kind: Pod
  metadata:
    creationTimestamp: "2026-01-15T18:40:11Z"
    generateName: classifier-deployment-68964c45b4-
    generation: 1
    labels:
      app: classifier
      pod-template-hash: 68964c45b4
    name: classifier-deployment-68964c45b4-zg4rr
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: ReplicaSet
      name: classifier-deployment-68964c45b4
      uid: 1d9d56f6-6c95-4580-a01f-993141a2abec
    resourceVersion: "1348"
    uid: d5141063-ec1d-49a6-8d95-65734cdd9fd8
  spec:
    containers:
    - image: classifier-k8s:latest
      imagePullPolicy: Never
      name: classifier
      ports:
      - containerPort: 7860
        protocol: TCP
      resources: {}
      terminationMessagePath: /dev/termination-log
      terminationMessagePolicy: File
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-2v5cb
        readOnly: true
    dnsPolicy: ClusterFirst
    enableServiceLinks: true
    nodeName: minikube
    preemptionPolicy: PreemptLowerPriority
    priority: 0
    restartPolicy: Always
    schedulerName: default-scheduler
    securityContext: {}
    serviceAccount: default
    serviceAccountName: default
    terminationGracePeriodSeconds: 30
    tolerations:
    - effect: NoExecute
      key: node.kubernetes.io/not-ready
      operator: Exists
      tolerationSeconds: 300
    - effect: NoExecute
      key: node.kubernetes.io/unreachable
      operator: Exists
      tolerationSeconds: 300
    volumes:
    - name: kube-api-access-2v5cb
      projected:
        defaultMode: 420
        sources:
        - serviceAccountToken:
            expirationSeconds: 3607
            path: token
        - configMap:
            items:
            - key: ca.crt
              path: ca.crt
            name: kube-root-ca.crt
        - downwardAPI:
            items:
            - fieldRef:
                apiVersion: v1
                fieldPath: metadata.namespace
              path: namespace
  status:
    conditions:
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:13Z"
      observedGeneration: 1
      status: "True"
      type: PodReadyToStartContainers
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:11Z"
      observedGeneration: 1
      status: "True"
      type: Initialized
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:13Z"
      observedGeneration: 1
      status: "True"
      type: Ready
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:13Z"
      observedGeneration: 1
      status: "True"
      type: ContainersReady
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:40:11Z"
      observedGeneration: 1
      status: "True"
      type: PodScheduled
    containerStatuses:
    - containerID: docker://a08434be5de542f60a78bd3b8cd73aa38fea112089df24540427722db229be2b
      image: classifier-k8s:latest
      imageID: docker://sha256:afbc229215b496350962c3dd646714031ea52ad2b8aaa34b79317f1044786cf8
      lastState: {}
      name: classifier
      ready: true
      resources: {}
      restartCount: 0
      started: true
      state:
        running:
          startedAt: "2026-01-15T18:40:12Z"
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-2v5cb
        readOnly: true
        recursiveReadOnly: Disabled
    hostIP: 192.168.49.2
    hostIPs:
    - ip: 192.168.49.2
    observedGeneration: 1
    phase: Running
    podIP: 10.244.0.6
    podIPs:
    - ip: 10.244.0.6
    qosClass: BestEffort
    startTime: "2026-01-15T18:40:11Z"
- apiVersion: v1
  kind: Service
  metadata:
    annotations:
      kubectl.kubernetes.io/last-applied-configuration: |
        {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"name":"classifier-service","namespace":"default"},"spec":{"ports":[{"port":80,"protocol":"TCP","targetPort":7860}],"selector":{"app":"classifier"}}}
    creationTimestamp: "2026-01-15T18:40:11Z"
    name: classifier-service
    namespace: default
    resourceVersion: "1319"
    uid: 7e82e2ff-1141-4eb5-a674-3e3e53924fcd
  spec:
    clusterIP: 10.101.19.55
    clusterIPs:
    - 10.101.19.55
    internalTrafficPolicy: Cluster
    ipFamilies:
    - IPv4
    ipFamilyPolicy: SingleStack
    ports:
    - port: 80
      protocol: TCP
      targetPort: 7860
    selector:
      app: classifier
    sessionAffinity: None
    type: ClusterIP
  status:
    loadBalancer: {}
- apiVersion: v1
  kind: Service
  metadata:
    creationTimestamp: "2026-01-15T18:23:30Z"
    labels:
      component: apiserver
      provider: kubernetes
    name: kubernetes
    namespace: default
    resourceVersion: "240"
    uid: 0f77c054-9e76-438f-9465-2c92953d1925
  spec:
    clusterIP: 10.96.0.1
    clusterIPs:
    - 10.96.0.1
    internalTrafficPolicy: Cluster
    ipFamilies:
    - IPv4
    ipFamilyPolicy: SingleStack
    ports:
    - name: https
      port: 443
      protocol: TCP
      targetPort: 8443
    sessionAffinity: None
    type: ClusterIP
  status:
    loadBalancer: {}
- apiVersion: apps/v1
  kind: Deployment
  metadata:
    annotations:
      deployment.kubernetes.io/revision: "1"
      kubectl.kubernetes.io/last-applied-configuration: |
        {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"app":"classifier"},"name":"classifier-deployment","namespace":"default"},"spec":{"replicas":2,"selector":{"matchLabels":{"app":"classifier"}},"template":{"metadata":{"labels":{"app":"classifier"}},"spec":{"containers":[{"image":"classifier-k8s:latest","imagePullPolicy":"Never","name":"classifier","ports":[{"containerPort":7860}]}]}}}}
    creationTimestamp: "2026-01-15T18:40:11Z"
    generation: 1
    labels:
      app: classifier
    name: classifier-deployment
    namespace: default
    resourceVersion: "1353"
    uid: 42c909b2-7d08-460f-ae99-8d072a7d4e12
  spec:
    progressDeadlineSeconds: 600
    replicas: 2
    revisionHistoryLimit: 10
    selector:
      matchLabels:
        app: classifier
    strategy:
      rollingUpdate:
        maxSurge: 25%
        maxUnavailable: 25%
      type: RollingUpdate
    template:
      metadata:
        labels:
          app: classifier
      spec:
        containers:
        - image: classifier-k8s:latest
          imagePullPolicy: Never
          name: classifier
          ports:
          - containerPort: 7860
            protocol: TCP
          resources: {}
          terminationMessagePath: /dev/termination-log
          terminationMessagePolicy: File
        dnsPolicy: ClusterFirst
        restartPolicy: Always
        schedulerName: default-scheduler
        securityContext: {}
        terminationGracePeriodSeconds: 30
  status:
    availableReplicas: 2
    conditions:
    - lastTransitionTime: "2026-01-15T18:40:13Z"
      lastUpdateTime: "2026-01-15T18:40:13Z"
      message: Deployment has minimum availability.
      reason: MinimumReplicasAvailable
      status: "True"
      type: Available
    - lastTransitionTime: "2026-01-15T18:40:11Z"
      lastUpdateTime: "2026-01-15T18:40:13Z"
      message: ReplicaSet "classifier-deployment-68964c45b4" has successfully progressed.
      reason: NewReplicaSetAvailable
      status: "True"
      type: Progressing
    observedGeneration: 1
    readyReplicas: 2
    replicas: 2
    updatedReplicas: 2
- apiVersion: apps/v1
  kind: ReplicaSet
  metadata:
    annotations:
      deployment.kubernetes.io/desired-replicas: "2"
      deployment.kubernetes.io/max-replicas: "3"
      deployment.kubernetes.io/revision: "1"
    creationTimestamp: "2026-01-15T18:40:11Z"
    generation: 1
    labels:
      app: classifier
      pod-template-hash: 68964c45b4
    name: classifier-deployment-68964c45b4
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: Deployment
      name: classifier-deployment
      uid: 42c909b2-7d08-460f-ae99-8d072a7d4e12
    resourceVersion: "1352"
    uid: 1d9d56f6-6c95-4580-a01f-993141a2abec
  spec:
    replicas: 2
    selector:
      matchLabels:
        app: classifier
        pod-template-hash: 68964c45b4
    template:
      metadata:
        labels:
          app: classifier
          pod-template-hash: 68964c45b4
      spec:
        containers:
        - image: classifier-k8s:latest
          imagePullPolicy: Never
          name: classifier
          ports:
          - containerPort: 7860
            protocol: TCP
          resources: {}
          terminationMessagePath: /dev/termination-log
          terminationMessagePolicy: File
        dnsPolicy: ClusterFirst
        restartPolicy: Always
        schedulerName: default-scheduler
        securityContext: {}
        terminationGracePeriodSeconds: 30
  status:
    availableReplicas: 2
    fullyLabeledReplicas: 2
    observedGeneration: 1
    readyReplicas: 2
    replicas: 2
kind: List
metadata:
  resourceVersion: ""
```
