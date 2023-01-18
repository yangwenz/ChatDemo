kubectl delete deployment chat-agent -n sfr-ns-wenzhuo-yang
kubectl delete deployment chat-backend -n sfr-ns-wenzhuo-yang
kubectl delete deployment chat-web -n sfr-ns-wenzhuo-yang
kubectl delete deployment redis-master -n sfr-ns-wenzhuo-yang

kubectl delete service chat-backend -n sfr-ns-wenzhuo-yang
kubectl delete service chat-web -n sfr-ns-wenzhuo-yang
kubectl delete service redis-master -n sfr-ns-wenzhuo-yang
