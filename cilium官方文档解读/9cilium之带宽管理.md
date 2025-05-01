我们直接来看Cilium的带宽管理器。它能干嘛？简单说，就是给你的**Pod网络流量做精细化管理，确保关键应用不被带宽洪流冲垮，同时也能给那些需要限速的Pod设定合理的速率**。这背后靠的是EDT和eBPF这两把刷子，技术上很硬核，效果也杠杠的。而且，它还是启用Pod级BBR拥塞控制的前提，后面会细说。

## Earliest Departure Time

先说说这个EDT，Earliest Departure Time。听起来有点绕，但核心思想是用时间戳来精确调度数据包，就像高铁调度一样，保证每个包都按最优时间点发出去，而不是傻乎乎地挤在一起。这样做的好处是啥？延迟低，吞吐量高，尤其是在高并发场景下，效果立竿见影。

再看eBPF，Extended Berkeley Packet Filter，扩展伯克利包过滤器。这玩意儿厉害了，直接在内核层面搞事情，处理数据包快到飞起，绕开了传统内核路径的层层关卡。对于需要实时监控、快速策略响应的网络场景，eBPF是效率神器。

可能有人会问，这不就是个带宽限制嘛，以前用CNI的带宽插件不行吗？还真不行。老的CNI带宽插件，像TBF Token Bucket Filter，用的是令牌桶算法，简单粗暴，但在多队列网络接口上，比如现在流行的25G、40G网卡，它就有点力不从心了，性能瓶颈明显。而且，随着集群规模越来越大，这种方案的扩展性就更难说了。

Cilium的带宽管理器就不一样了，它基于EDT和eBPF，是原生集成到Cilium里的，不是那种链式挂载的插件，天生就更适合高性能、高并发的场景，尤其是在大规模集群里，优势非常明显。这里要特别强调一个点，就是BPF Host Routing。Cilium的官方文档强烈建议，**带宽管理器必须和BPF Host Routing一起用**。为什么？因为如果你不用BPF Host Routing，数据包在网络栈里走的路径就比较迂回，可能会经过一些传统的转发、路由层，这些层处理起来比较慢，就会引入不必要的延迟。尤其是在需要低延迟的场景下，这个延迟可能就是致命的。所以，为了保证带宽管理器的性能，特别是低延迟，强烈建议搭配BPF Host Routing。具体的数据对比可以参考原文的链接，里面有更详细的分析。

怎么用这个带宽管理器呢？很简单，通过Kubernetes的Pod Annotation来设置。你可以给你的Pod打上一个叫kubernetes.io/egress-bandwidth的标签，比如设置成10M，意思就是限制这个Pod的出站带宽为10兆比特每秒。这个限制是在主机的网络设备出站口生效的，无论是直接路由还是隧道模式，都支持。

但是要注意，入站带宽限制，也就是kubernetes.io/ingress-bandwidth，Cilium是不支持的，也不推荐用。为什么？因为入站限制是在数据包进来的时候就在ifb设备上做处理，相当于在数据包刚到节点，还没上栈的时候就加了一道缓冲，这会增加额外的延迟，而且这时候带宽已经占用了，节点也处理了这个包，再做限制就有点晚了。所以，只管出站。

配置起来也很方便。如果你用的是Helm，那几步就搞定。

```
helm install cilium cilium/cilium --version 1.17.3 \
  --namespace kube-system \
  --set bandwidthManager.enabled=true
```

安装Cilium的时候，加上一个参数，--set bandwidthManager.enabled=true，这样就能把带宽管理器给启用了。当然，如果你觉得自动检测不够灵活，或者你的网络环境比较特殊，也可以手动指定设备。通过helm的devices选项，你可以列出具体的设备名，比如devices等于eth0逗号eth1逗号eth2。但是要注意，如果你手动指定了，那么在所有Cilium管理的节点上，这个设备的名称必须是完全一致的，否则可能会有问题。

配置完了，怎么知道它是不是真的启动了呢？很简单。首先，检查一下Cilium的Pod是不是都正常运行了，可以用

```
$ kubectl -n kube-system exec ds/cilium -- cilium-dbg status | grep BandwidthManager
BandwidthManager:       EDT with BPF [BBR] [eth0]
```

如果一切正常，你会看到类似BandwidthManager: EDT with BPF [BBR] [eth0]这样的输出。这说明带宽管理器已经启用，而且正在使用EDT和BPF，如果也开启了BBR，还会显示出来。eth0就是刚才检测到的那个需要应用带宽限制的设备。

理论讲完了，来点实际的。我们用netperf工具来验证一下这个带宽限制到底有没有效果。我们先部署两个Pod：一个叫netperf-server，给它打上kubernetes.io/egress-bandwidth等于10M的标签，限制它出站带宽为10Mbit/s；另一个叫netperf-client，用来做测试。

```yml
---
apiVersion: v1
kind: Pod
metadata:
  annotations:
    # Limits egress bandwidth to 10Mbit/s.
    kubernetes.io/egress-bandwidth: "10M"
  labels:
    # This pod will act as server.
    app.kubernetes.io/name: netperf-server
  name: netperf-server
spec:
  containers:
  - name: netperf
    image: cilium/netperf
    ports:
    - containerPort: 12865
---
apiVersion: v1
kind: Pod
metadata:
  # This Pod will act as client.
  name: netperf-client
spec:
  affinity:
    # Prevents the client from being scheduled to the
    # same node as the server.
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - netperf-server
        topologyKey: kubernetes.io/hostname
  containers:
  - name: netperf
    args:
    - sleep
    - infinity
    image: cilium/netperf
```

为了让测试更公平，我们用podAntiAffinity确保这两个Pod不在同一个节点上。然后，我们用netperf-client去测试netperf-server的带宽。因为是从服务器端流向客户端，所以我们要用TCP_MAERTS这个测试。执行命令后，你会看到类似这样的结果：Throughput 10.00 9.56。看到没？实际吞吐量9.56Mbit/s，非常接近我们设定的10Mbit/s，这说明带宽限制是实实在在生效的。

```shell
$ NETPERF_SERVER_IP=$(kubectl get pod netperf-server -o jsonpath='{.status.podIP}')
$ kubectl exec netperf-client -- \
    netperf -t TCP_MAERTS -H "${NETPERF_SERVER_IP}"
MIGRATED TCP MAERTS TEST from 0.0.0.0 (0.0.0.0) port 0 AF_INET to 10.217.0.254 () port 0 AF_INET
Recv   Send    Send
Socket Socket  Message  Elapsed
Size   Size    Size     Time     Throughput
bytes  bytes   bytes    secs.    10^6bits/sec

 87380  16384  16384    10.00       9.56
```

如果想更深入地了解，看看BPF侧是怎么设置这个带宽限制的，可以使用cilium-dbg bpf bandwidth list命令。这个命令需要在Cilium的Pod里执行，比如 cilium-xxxxx，你需要找到和你那个被限制的Pod co-located的那个Cilium Pod。执行结果会列出每个Endpoint的标识符和对应的出站带宽限制。比如，你会看到类似这样的输出：IDENTITY EGRESS BANDWIDTH 491 10M。这里的491就是这个Pod的Endpoint标识，你可以把它和cilium-dbg endpoint list命令的输出关联起来，找到对应的Pod。这样，你就能精确地看到每个Pod的带宽限制是多少。

```
$ kubectl exec -it -n kube-system cilium-xxxxxx -- cilium-dbg bpf bandwidth list
IDENTITY   EGRESS BANDWIDTH (BitsPerSec)
491        10M
```

## BBR

前面提到了BBR，现在我们来聊聊它。BBR，全称Bottleneck Bandwidth and Round-trip time，瓶颈带宽和往返时间。这玩意儿是谷歌搞出来的，号称是下一代拥塞控制算法。它跟传统的TCP拥塞控制算法比如CUBIC相比，最大的优势就是能提供更高的带宽和更低的延迟，尤其是在互联网这种复杂的网络环境下。有多厉害？数据说话：吞吐量可以比最好的基于丢包的拥塞控制算法高出2700倍，排队延迟可以低到25倍！这意味着什么？意味着你的应用响应更快，用户体验更好，尤其是在那些直接对外提供服务的Pod上，效果非常显著。

所以，如果你的集群里有Pod需要直接暴露给外部互联网用户，那BBR绝对是你的不二之选。启用BBR也很简单，前提是你的Linux内核版本要达到5.18或者更高。这个版本要求是因为早期的内核在处理Pod到宿主机网络命名空间切换时，会丢失一些关键的时间戳信息，导致BBR无法正常工作。我们解决了这个问题，所以现在BBR才能在Pod里稳定运行。启用BBR的命令和启用带宽管理器类似，只是在helm upgrade命令里多加一个参数--set bandwidthManager.bbr=true。

```
helm upgrade cilium cilium/cilium --version 1.17.3 \
  --namespace kube-system \
  --reuse-values \
  --set bandwidthManager.enabled=true \
  --set bandwidthManager.bbr=true
kubectl -n kube-system rollout restart ds/cilium
```

执行完这个命令，重启Cilium，然后所有新创建的Pod都会默认使用BBR了。为什么BBR需要这么高的内核版本？主要是因为早期的内核在处理网络命名空间切换时，会丢失一些关键的时间戳信息，而这些信息对于BBR的精确计算至关重要。我们和Linux社区合作，修复了这个问题，使得BBR在Pod里也能正常工作。另外，BBR还需要BPF Host Routing的支持。因为BBR需要追踪数据包的socket关联，一直追踪到物理设备上的FQ队列，才能进行精确的流量控制和拥塞探测。如果没有BPF Host Routing，数据包在宿主机的转发层就会丢失socket信息，BBR就无法正常工作。这也是为什么我们强烈推荐带宽管理器和BPF Host Routing一起使用的原因。

启用BBR后，怎么验证它是否成功应用了呢？其实还是看cilium-dbg status的输出。

```
$ kubectl -n kube-system exec ds/cilium -- cilium-dbg status | grep BandwidthManager
BandwidthManager:       EDT with BPF [BBR] [eth0]
```

如果看到输出里带了BBR，比如 BandwidthManager: EDT with BPF [BBR] [eth0]，这就说明BBR已经成功启用并应用到你的网络设备eth0上了。这个输出信息会告诉你，当前的带宽管理器是基于EDT和BPF实现的，而且支持BBR，作用在eth0设备上。这样，你就可以放心地享受BBR带来的性能提升啦。关于BBR的使用，有几个最佳实践要分享。

- 首先，强烈建议在集群初始化的时候就启用BBR，这样整个集群的所有节点和Pod都能统一使用BBR。如果一部分Pod用的是CUBIC，一部分用的是BBR，那么在同一个网络环境下，可能会出现不公平性，导致性能波动。
- 其次，BBR由于其探测机制，可能会比CUBIC产生更多的TCP重传，这在某些情况下是正常的
- 最后，再次强调，BBR最适合那些需要直接面向外部互联网客户端的集群，比如Web应用、API服务等。对于内部服务之间的通信，可能CUBIC的性能也足够了。

当然，任何技术都不是完美的，Cilium的带宽管理器也有一些局限性。

- 第一个是，它目前和L7层的Cilium Network Policies存在冲突。如果你的L7策略恰好在出站方向拦截了某个Pod的流量，那么这个Pod的带宽限制就会失效。这主要是因为L7策略在处理时会绕过底层的网络包处理，而带宽管理器是在网络包层面进行控制的。
- 第二个是，它在嵌套的网络命名空间环境下，比如Kind这种轻量级的Kubernetes环境，可能无法正常工作。这是因为嵌套环境通常无法访问到全局的sysctl参数，而带宽管理器的实现依赖于这些参数。所以，在使用这些特殊环境时，需要注意这些限制。

总结一下，Cilium的带宽管理器，凭借其基于EDT和eBPF的技术，以及对BBR的支持，为我们提供了一个高效、精准、灵活的网络带宽管理工具。它不仅能帮助我们优化TCP和UDP流量，还能显著提升面向互联网的Pod的性能。未来，随着技术的不断发展，相信它会越来越强大，成为云原生网络环境中不可或缺的利器。