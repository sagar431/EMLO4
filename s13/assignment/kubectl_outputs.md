# Kubectl Outputs

Generated on: Thu Jan 15 18:58:48 UTC 2026

## Deployment Description
```
Name:                   dog-classifier-deployment
Namespace:              default
CreationTimestamp:      Thu, 15 Jan 2026 18:56:57 +0000
Labels:                 app=dog-classifier
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               app=dog-classifier
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
Pod Template:
  Labels:  app=dog-classifier
  Containers:
   dog-classifier:
    Image:      dog-classifier:latest
    Port:       8000/TCP
    Host Port:  0/TCP
    Limits:
      cpu:     1
      memory:  2Gi
    Requests:
      cpu:         250m
      memory:      512Mi
    Liveness:      http-get http://:8000/health delay=60s timeout=1s period=10s #success=1 #failure=3
    Readiness:     http-get http://:8000/health delay=30s timeout=1s period=5s #success=1 #failure=3
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
NewReplicaSet:   dog-classifier-deployment-6b699598f5 (2/2 replicas created)
Events:
  Type    Reason             Age   From                   Message
  ----    ------             ----  ----                   -------
  Normal  ScalingReplicaSet  112s  deployment-controller  Scaled up replica set dog-classifier-deployment-6b699598f5 from 0 to 2
```

## Pod Description
```
Name:             dog-classifier-deployment-6b699598f5-gzksk
Namespace:        default
Priority:         0
Service Account:  default
Node:             minikube/192.168.49.2
Start Time:       Thu, 15 Jan 2026 18:56:57 +0000
Labels:           app=dog-classifier
                  pod-template-hash=6b699598f5
Annotations:      <none>
Status:           Running
IP:               10.244.0.13
IPs:
  IP:           10.244.0.13
Controlled By:  ReplicaSet/dog-classifier-deployment-6b699598f5
Containers:
  dog-classifier:
    Container ID:   docker://91ab8932da80b72ba4a45c087fb0cd048ba77874e74fb04501ef996582a92f74
    Image:          dog-classifier:latest
    Image ID:       docker://sha256:3f846127ac127eb0762a8f52e719833769447dd26df073a1a8597a2c6775b5a6
    Port:           8000/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Thu, 15 Jan 2026 18:56:58 +0000
    Ready:          True
    Restart Count:  0
    Limits:
      cpu:     1
      memory:  2Gi
    Requests:
      cpu:        250m
      memory:     512Mi
    Liveness:     http-get http://:8000/health delay=60s timeout=1s period=10s #success=1 #failure=3
    Readiness:    http-get http://:8000/health delay=30s timeout=1s period=5s #success=1 #failure=3
    Environment:  <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-wrtfm (ro)
Conditions:
  Type                        Status
  PodReadyToStartContainers   True 
  Initialized                 True 
  Ready                       True 
  ContainersReady             True 
  PodScheduled                True 
Volumes:
  kube-api-access-wrtfm:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    Optional:                false
    DownwardAPI:             true
QoS Class:                   Burstable
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  112s  default-scheduler  Successfully assigned default/dog-classifier-deployment-6b699598f5-gzksk to minikube
  Normal  Pulled     111s  kubelet            spec.containers{dog-classifier}: Container image "dog-classifier:latest" already present on machine
  Normal  Created    111s  kubelet            spec.containers{dog-classifier}: Created container: dog-classifier
  Normal  Started    111s  kubelet            spec.containers{dog-classifier}: Started container dog-classifier
```

## Ingress Description
```
Name:             dog-classifier-ingress
Labels:           <none>
Namespace:        default
Address:          192.168.49.2
Ingress Class:    nginx
Default backend:  <default>
Rules:
  Host                      Path  Backends
  ----                      ----  --------
  dog-classifier.localhost  
                            /   dog-classifier-service:80 (10.244.0.13:8000,10.244.0.14:8000)
Annotations:                nginx.ingress.kubernetes.io/affinity: cookie
                            nginx.ingress.kubernetes.io/affinity-mode: balanced
                            nginx.ingress.kubernetes.io/proxy-body-size: 50m
                            nginx.ingress.kubernetes.io/session-cookie-expires: 172800
                            nginx.ingress.kubernetes.io/session-cookie-max-age: 172800
                            nginx.ingress.kubernetes.io/session-cookie-name: INGRESSCOOKIE
Events:
  Type    Reason  Age                    From                      Message
  ----    ------  ----                   ----                      -------
  Normal  Sync    7m41s (x2 over 8m35s)  nginx-ingress-controller  Scheduled for sync
```

## Pod Metrics (kubectl top pod)
```
NAME                                         CPU(cores)   MEMORY(bytes)   
dog-classifier-deployment-6b699598f5-gzksk   90m          550Mi           
dog-classifier-deployment-6b699598f5-sbnk5   151m         538Mi           
```

## Node Metrics (kubectl top node)
```
NAME       CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)   
minikube   621m         2%       2706Mi          1%          
```

## All Resources (kubectl get all -o yaml)
```yaml
apiVersion: v1
items:
- apiVersion: v1
  kind: Pod
  metadata:
    creationTimestamp: "2026-01-15T18:56:57Z"
    generateName: dog-classifier-deployment-6b699598f5-
    generation: 1
    labels:
      app: dog-classifier
      pod-template-hash: 6b699598f5
    name: dog-classifier-deployment-6b699598f5-gzksk
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: ReplicaSet
      name: dog-classifier-deployment-6b699598f5
      uid: 495e660e-d56b-4deb-8cf5-788802196da4
    resourceVersion: "2766"
    uid: 8558dc9c-c556-4117-a826-8a21d48cf066
  spec:
    containers:
    - image: dog-classifier:latest
      imagePullPolicy: Never
      livenessProbe:
        failureThreshold: 3
        httpGet:
          path: /health
          port: 8000
          scheme: HTTP
        initialDelaySeconds: 60
        periodSeconds: 10
        successThreshold: 1
        timeoutSeconds: 1
      name: dog-classifier
      ports:
      - containerPort: 8000
        protocol: TCP
      readinessProbe:
        failureThreshold: 3
        httpGet:
          path: /health
          port: 8000
          scheme: HTTP
        initialDelaySeconds: 30
        periodSeconds: 5
        successThreshold: 1
        timeoutSeconds: 1
      resources:
        limits:
          cpu: "1"
          memory: 2Gi
        requests:
          cpu: 250m
          memory: 512Mi
      terminationMessagePath: /dev/termination-log
      terminationMessagePolicy: File
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-wrtfm
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
    - name: kube-api-access-wrtfm
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
      lastTransitionTime: "2026-01-15T18:56:59Z"
      observedGeneration: 1
      status: "True"
      type: PodReadyToStartContainers
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:56:57Z"
      observedGeneration: 1
      status: "True"
      type: Initialized
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:57:30Z"
      observedGeneration: 1
      status: "True"
      type: Ready
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:57:30Z"
      observedGeneration: 1
      status: "True"
      type: ContainersReady
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:56:57Z"
      observedGeneration: 1
      status: "True"
      type: PodScheduled
    containerStatuses:
    - allocatedResources:
        cpu: 250m
        memory: 512Mi
      containerID: docker://91ab8932da80b72ba4a45c087fb0cd048ba77874e74fb04501ef996582a92f74
      image: dog-classifier:latest
      imageID: docker://sha256:3f846127ac127eb0762a8f52e719833769447dd26df073a1a8597a2c6775b5a6
      lastState: {}
      name: dog-classifier
      ready: true
      resources:
        limits:
          cpu: "1"
          memory: 2Gi
        requests:
          cpu: 250m
          memory: 512Mi
      restartCount: 0
      started: true
      state:
        running:
          startedAt: "2026-01-15T18:56:58Z"
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-wrtfm
        readOnly: true
        recursiveReadOnly: Disabled
    hostIP: 192.168.49.2
    hostIPs:
    - ip: 192.168.49.2
    observedGeneration: 1
    phase: Running
    podIP: 10.244.0.13
    podIPs:
    - ip: 10.244.0.13
    qosClass: Burstable
    startTime: "2026-01-15T18:56:57Z"
- apiVersion: v1
  kind: Pod
  metadata:
    creationTimestamp: "2026-01-15T18:56:57Z"
    generateName: dog-classifier-deployment-6b699598f5-
    generation: 1
    labels:
      app: dog-classifier
      pod-template-hash: 6b699598f5
    name: dog-classifier-deployment-6b699598f5-sbnk5
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: ReplicaSet
      name: dog-classifier-deployment-6b699598f5
      uid: 495e660e-d56b-4deb-8cf5-788802196da4
    resourceVersion: "2770"
    uid: b0936674-a2f8-4063-9ed2-a3313c1f940b
  spec:
    containers:
    - image: dog-classifier:latest
      imagePullPolicy: Never
      livenessProbe:
        failureThreshold: 3
        httpGet:
          path: /health
          port: 8000
          scheme: HTTP
        initialDelaySeconds: 60
        periodSeconds: 10
        successThreshold: 1
        timeoutSeconds: 1
      name: dog-classifier
      ports:
      - containerPort: 8000
        protocol: TCP
      readinessProbe:
        failureThreshold: 3
        httpGet:
          path: /health
          port: 8000
          scheme: HTTP
        initialDelaySeconds: 30
        periodSeconds: 5
        successThreshold: 1
        timeoutSeconds: 1
      resources:
        limits:
          cpu: "1"
          memory: 2Gi
        requests:
          cpu: 250m
          memory: 512Mi
      terminationMessagePath: /dev/termination-log
      terminationMessagePolicy: File
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-phfk9
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
    - name: kube-api-access-phfk9
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
      lastTransitionTime: "2026-01-15T18:56:59Z"
      observedGeneration: 1
      status: "True"
      type: PodReadyToStartContainers
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:56:57Z"
      observedGeneration: 1
      status: "True"
      type: Initialized
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:57:30Z"
      observedGeneration: 1
      status: "True"
      type: Ready
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:57:30Z"
      observedGeneration: 1
      status: "True"
      type: ContainersReady
    - lastProbeTime: null
      lastTransitionTime: "2026-01-15T18:56:57Z"
      observedGeneration: 1
      status: "True"
      type: PodScheduled
    containerStatuses:
    - allocatedResources:
        cpu: 250m
        memory: 512Mi
      containerID: docker://485b9de8913c5e6c3c1caedc862a98706b941241dc4a0549b1634860ef3cc8d2
      image: dog-classifier:latest
      imageID: docker://sha256:3f846127ac127eb0762a8f52e719833769447dd26df073a1a8597a2c6775b5a6
      lastState: {}
      name: dog-classifier
      ready: true
      resources:
        limits:
          cpu: "1"
          memory: 2Gi
        requests:
          cpu: 250m
          memory: 512Mi
      restartCount: 0
      started: true
      state:
        running:
          startedAt: "2026-01-15T18:56:58Z"
      volumeMounts:
      - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
        name: kube-api-access-phfk9
        readOnly: true
        recursiveReadOnly: Disabled
    hostIP: 192.168.49.2
    hostIPs:
    - ip: 192.168.49.2
    observedGeneration: 1
    phase: Running
    podIP: 10.244.0.14
    podIPs:
    - ip: 10.244.0.14
    qosClass: Burstable
    startTime: "2026-01-15T18:56:57Z"
- apiVersion: v1
  kind: Service
  metadata:
    annotations:
      kubectl.kubernetes.io/last-applied-configuration: |
        {"apiVersion":"v1","kind":"Service","metadata":{"annotations":{},"name":"dog-classifier-service","namespace":"default"},"spec":{"ports":[{"port":80,"protocol":"TCP","targetPort":8000}],"selector":{"app":"dog-classifier"},"type":"ClusterIP"}}
    creationTimestamp: "2026-01-15T18:50:14Z"
    name: dog-classifier-service
    namespace: default
    resourceVersion: "2002"
    uid: c07f2854-5a0a-45fe-b52d-d8316f34b7a5
  spec:
    clusterIP: 10.98.66.246
    clusterIPs:
    - 10.98.66.246
    internalTrafficPolicy: Cluster
    ipFamilies:
    - IPv4
    ipFamilyPolicy: SingleStack
    ports:
    - port: 80
      protocol: TCP
      targetPort: 8000
    selector:
      app: dog-classifier
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
        {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"app":"dog-classifier"},"name":"dog-classifier-deployment","namespace":"default"},"spec":{"replicas":2,"selector":{"matchLabels":{"app":"dog-classifier"}},"template":{"metadata":{"labels":{"app":"dog-classifier"}},"spec":{"containers":[{"image":"dog-classifier:latest","imagePullPolicy":"Never","livenessProbe":{"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":60,"periodSeconds":10},"name":"dog-classifier","ports":[{"containerPort":8000}],"readinessProbe":{"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":30,"periodSeconds":5},"resources":{"limits":{"cpu":"1000m","memory":"2Gi"},"requests":{"cpu":"250m","memory":"512Mi"}}}]}}}}
    creationTimestamp: "2026-01-15T18:56:57Z"
    generation: 1
    labels:
      app: dog-classifier
    name: dog-classifier-deployment
    namespace: default
    resourceVersion: "2775"
    uid: c3678dc6-8d4c-4780-b5fa-c444434255a8
  spec:
    progressDeadlineSeconds: 600
    replicas: 2
    revisionHistoryLimit: 10
    selector:
      matchLabels:
        app: dog-classifier
    strategy:
      rollingUpdate:
        maxSurge: 25%
        maxUnavailable: 25%
      type: RollingUpdate
    template:
      metadata:
        labels:
          app: dog-classifier
      spec:
        containers:
        - image: dog-classifier:latest
          imagePullPolicy: Never
          livenessProbe:
            failureThreshold: 3
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 60
            periodSeconds: 10
            successThreshold: 1
            timeoutSeconds: 1
          name: dog-classifier
          ports:
          - containerPort: 8000
            protocol: TCP
          readinessProbe:
            failureThreshold: 3
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 30
            periodSeconds: 5
            successThreshold: 1
            timeoutSeconds: 1
          resources:
            limits:
              cpu: "1"
              memory: 2Gi
            requests:
              cpu: 250m
              memory: 512Mi
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
    - lastTransitionTime: "2026-01-15T18:57:30Z"
      lastUpdateTime: "2026-01-15T18:57:30Z"
      message: Deployment has minimum availability.
      reason: MinimumReplicasAvailable
      status: "True"
      type: Available
    - lastTransitionTime: "2026-01-15T18:56:57Z"
      lastUpdateTime: "2026-01-15T18:57:30Z"
      message: ReplicaSet "dog-classifier-deployment-6b699598f5" has successfully
        progressed.
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
    creationTimestamp: "2026-01-15T18:56:57Z"
    generation: 1
    labels:
      app: dog-classifier
      pod-template-hash: 6b699598f5
    name: dog-classifier-deployment-6b699598f5
    namespace: default
    ownerReferences:
    - apiVersion: apps/v1
      blockOwnerDeletion: true
      controller: true
      kind: Deployment
      name: dog-classifier-deployment
      uid: c3678dc6-8d4c-4780-b5fa-c444434255a8
    resourceVersion: "2774"
    uid: 495e660e-d56b-4deb-8cf5-788802196da4
  spec:
    replicas: 2
    selector:
      matchLabels:
        app: dog-classifier
        pod-template-hash: 6b699598f5
    template:
      metadata:
        labels:
          app: dog-classifier
          pod-template-hash: 6b699598f5
      spec:
        containers:
        - image: dog-classifier:latest
          imagePullPolicy: Never
          livenessProbe:
            failureThreshold: 3
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 60
            periodSeconds: 10
            successThreshold: 1
            timeoutSeconds: 1
          name: dog-classifier
          ports:
          - containerPort: 8000
            protocol: TCP
          readinessProbe:
            failureThreshold: 3
            httpGet:
              path: /health
              port: 8000
              scheme: HTTP
            initialDelaySeconds: 30
            periodSeconds: 5
            successThreshold: 1
            timeoutSeconds: 1
          resources:
            limits:
              cpu: "1"
              memory: 2Gi
            requests:
              cpu: 250m
              memory: 512Mi
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
