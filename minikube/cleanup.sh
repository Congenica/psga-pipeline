#!/bin/bash


## delete DB
kubectl delete deployment psql
kubectl delete service psql-host
kubectl delete pvc psql-pvc


## delete covid-pipeline
kubectl delete deployment covid-pipeline
kubectl delete pvc covid-pipeline-pvc


## delete any nf pod
#kubectl get pods -n default --no-headers=true | awk '/nf/{print $1}'| xargs  kubectl delete -n default pod
