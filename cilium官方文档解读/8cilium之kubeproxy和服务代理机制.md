要确认HostPort功能已经成功启用，最直接的方法就是使用 cilium-dbg status 这个命令。这个命令会显示 kubeProxyReplacement 的详细信息，其中 HostPort: Enabled 这个关键词就是我们确认的信号。如果看到这个，就说明 HostPort 功能已经就位了。

理论验证完了，我们来动手实践一下。

- 修改示例 YAML 文件，添加 `hostPort: 8080` 参数
- 部署修改后的 Deployment
- 验证 Cilium eBPF 替代方案是否正确暴露 HostPort
- 检查 `cilium-dbg service list` 输出
- 验证 iptables 中不存在 HostPort 服务规则
- `curl` 测试节点 IP 的 HostPort 端口是否可达

这里有个修改后的示例 YAML 文件，关键在于给容器加了一个 hostPort: 8080 的参数。

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-nginx
spec:
  selector:
    matchLabels:
      run: my-nginx
  replicas: 1
  template:
    metadata:
      labels:
        run: my-nginx
    spec:
      containers:
      - name: my-nginx
        image: nginx
        ports:
        - containerPort: 80
          hostPort: 8080
```

部署这个 Deployment 之后，我们就要检查 Cilium 的 eBPF 替代方案是否真的把容器端口 80 暴露到了宿主机的 8080 端口。怎么查？用 cilium-dbg service list。如果能看到类似 192.168.178.29:8080 对应到 10.29.207.199:80 的记录，就说明映射成功了。同时，为了排除 iptables 的干扰，我们还要检查一下主机的 iptables 规则，确保没有针对这个 HostPort 的规则。最后一步，也是最直观的，就是用 curl 测试一下，从节点 IP 加上 8080 端口访问，看看是不是能正常返回 nginx 的欢迎页面。这三步走下来，HostPort 功能的验证就比较完整了。当然，一个功能的健壮性体现在它的生命周期管理上。我们刚刚部署了 my-nginx，然后把它删掉。这时候，再运行 cilium-dbg service list，你会发现之前那个 192.168.178.29:8080 的 HostPort 条目，它消失了。这说明什么？说明 Cilium 能够正确地管理 HostPort 的生命周期，当对应的 Pod 被删除时，相关的 HostPort 配置也会随之消失，不会留下无用的规则或者残留的资源。这对于维护集群的干净和高效至关重要。

## kube-proxy Hybrid Modes

现在我们来聊聊 Cilium 的 kube-proxy 替代方案的部署模式。它不是非黑即白的要么全替代，要么完全不用。Cilium 提供了灵活的混合模式。你可以选择让 Cilium 完全取代 kube-proxy，实现无 kube-proxy 的环境，这通常是推荐的方案。但如果你的 Linux 内核版本不支持某些功能，或者出于其他原因，你也可以选择让 Cilium 和 kube-proxy 共存。不过要注意，这种共存模式下，两个机制是独立运作的，它们的 NAT 表格互不相干。这意味着，如果你在运行时切换模式，比如从 Cilium 切回 kube-proxy，或者反过来，现有的连接很可能会中断，因为两边的规则都不一样了。所以，如果你是在一个已有的、正在运行的集群上尝试共存模式，务必提前做好准备。

我们先来看最推荐的模式：

kubeProxyReplacement等于true。这意味着 Cilium 将全面接管 Kubernetes 的服务网络功能，包括 ClusterIP、NodePort、LoadBalancer、externalIPs 以及我们刚刚讨论的 HostPort。理想情况下，你应该在没有 kube-proxy 的环境中运行 Kubernetes，让 Cilium 成为唯一的网络服务代理。当然，如果因为某些原因，比如 Kubernetes 发行版的限制，你不得不保留 kube-proxy，那也可以接受，但要记住共存模式的潜在风险。一旦 Cilium Agent 启动，它就会负责处理这些服务类型。如果底层的内核版本不支持，Cilium Agent 会直接报错，阻止启动。

另一种模式是 kubeProxyReplacement等于false。这表示 Cilium 完全放弃服务网络的管理，一切交给 kube-proxy。这种模式通常只在特定场景下使用，比如需要进行混合部署，或者在某些内核不支持的情况下，作为临时方案。在这种模式下，你需要手动启用 Cilium 的 eBPF 替代组件，比如 socketLB、nodePort、externalIPs、hostPort 这些。默认情况下，它们都是关闭的，你需要根据需要把它们设为 true。特别要注意，如果你启用 nodePort.enabled，一定要同时把 nodePort.enableHealthcheck 设为 false，否则 Cilium 和 kube-proxy 都会尝试启动 NodePort 的健康检查服务器，导致端口冲突。同样，切换模式或者启用这些组件，都可能影响现有连接，建议在操作前做好流量调度。

为了让大家更直观地理解 false 模式下的配置，这里有几个 Helm 安装的示例。第一个例子，虽然设置了 false，但通过手动启用 socketLB、nodePort、externalIPs 和 hostPort，实际上效果等同于 true 模式，适用于完全无 kube-proxy 的环境。

```shell
helm install cilium cilium/cilium --version 1.17.3 \
--namespace kube-system \
--set kubeProxyReplacement=false \
--set socketLB.enabled=true \
--set nodePort.enabled=true \
--set externalIPs.enabled=true \
--set hostPort.enabled=true \
--set k8sServiceHost=${API_SERVER_IP} \
--set k8sServicePort=${API_SERVER_PORT}
```

第二个例子，只设置了 false，这相当于 Cilium 1.6 或更早版本在有 kube-proxy 环境下的默认行为，主要负责 Pod 之间的 ClusterIP 服务。

```
helm install cilium cilium/cilium --version 1.17.3 \
--namespace kube-system \
--set kubeProxyReplacement=false 
```

第三个例子，只启用了 nodePort 和 externalIPs，这是在有 kube-proxy 的环境中，Cilium 用来优化 NodePort 和外部服务处理的典型配置。

```shell
helm install cilium cilium/cilium --version 1.17.3 \
--namespace kube-system \
--set kubeProxyReplacement=false \
--set nodePort.enabled=true \
--set externalIPs.enabled=true
```

这些例子展示了如何通过精细控制来实现不同的混合部署策略。无论你选择了哪种模式，都需要一种方法来确认当前的配置状态。幸运的是，我们之前提到的 cilium-dbg status 命令也能帮我们做到这一点。通过 grep 命令筛选出 kubeProxyReplacement 的信息，就能看到当前 Cilium 是处于 True 还是 False 模式。比如，如果输出显示 True，后面可能还会跟着像 eth0 (DR) 这样的信息，表示它使用的是直接路由模式。这样就能清晰地知道 Cilium 当前是如何与 kube-proxy 或者说它自己替代的 kube-proxy 交互的。

## Graceful Termination

接下来我们看看优雅终止。这个特性对于保证服务的平稳过渡非常重要。当你的服务 Pod 需要被替换或删除时，你希望它能优雅地结束，而不是突然中断。Cilium 的 eBPF 替代方案支持这一点。不过，这个功能需要 Kubernetes 1.20 或更高版本，而且要启用 EndpointSliceTerminatingcondition 这个特性门。

默认情况下，这个功能是开启的，但如果你的场景不需要，可以通过配置项 enable-k8s-terminating-endpoint 来关闭。同样，我们可以通过 cilium-dbg status --verbose 来查看这个功能是否启用，找到 Graceful Termination: Enabled 这样的信息。那么，优雅终止具体是怎么工作的呢？

- 当 Cilium Agent 检测到一个 Pod 进入终止状态时，它会移除这个 Pod 的数据路径状态，这意味着它不再接受新的连接请求。
- 对于已经建立的连接，它会允许它们继续完成，直到自然关闭。
- 只有当 Kubernetes 发送 Pod 的删除事件时，Cilium 才会彻底移除这个 Pod 的状态。

这个过程和 Kubernetes 的 terminationGracePeriodSeconds 参数配合，共同控制了 Pod 的优雅退出时间。当然，还有一些特殊场景，比如零中断滚动更新，可能需要在 Pod 的终止期间仍然允许流量发送到它，这涉及到 Kubernetes 的流量工程策略，大家可以参考相关的 Kubernetes 官方文档。

## Session Affinity

会话亲和性是另一个重要的服务特性。简单来说，就是让同一个客户端的请求总是被路由到同一个后端服务。这对于需要保持状态的应用非常重要。

Cilium 通过 sessionAffinity: ClientIP 来实现这个功能。默认情况下，亲和性的有效期是 3 小时，但你可以通过 Kubernetes 的 sessionAffinityConfig 来调整。亲和性的来源，也就是怎么判断客户端，取决于请求的来源。

- 如果是从集群外部来的，就用源 IP 地址。
- 如果是从集群内部来的，那就复杂一点了。
  - 如果启用了 socket-LB，也就是在 socket 层做负载均衡
    - 那么会用到客户端的网络命名空间 cookie，这个是 Linux 内核 5.7 之后引入的特性，因为 socket 层还没形成包，没法拿到 IP 地址。如果不使用 socket-LB，那就还是老老实实用源 IP 地址。默认情况下，这个功能是开启的，
    - 但如果你的内核版本比较旧，不支持网络命名空间 cookie，Cilium 会退回到一个基于固定 cookie 值的模式，虽然有点 trade-off，但也能保证基本的亲和性。

对于多端口服务，亲和性是按服务 IP 和端口分开算的。另外，如果同时用了 Maglev 哈希算法选后端，Maglev 会忽略源端口，以尊重用户设置的 ClientIP 亲和性。

## Health Check

健康检查是确保服务可用性的关键。Cilium 的 eBPF 替代方案也提供了健康检查服务。不过，这个功能是默认关闭的。如果你想启用它，需要配置一个选项，叫做 kubeProxyReplacementHealthzBindAddr。这个选项需要指定一个 IP 地址和端口号，告诉 Cilium 健康检查服务器监听在哪里。比如，你想让 IPv4 的健康检查服务器监听所有网卡的 10256 端口，就配置成 0.0.0.0:10256。对于 IPv6，就用 方括号冒号冒号 方括号。配置好之后，你就可以通过访问 HTTP 的斜杠healthz 端点来检查服务的健康状况了。

## Source Ranges Checks

LoadBalancer 服务的安全性是大家关心的问题。Cilium 提供了 LoadBalancerSourceRanges 检查功能，可以让你白名单允许访问 LoadBalancer 服务的源 IP 地址段。如果你配置了这个字段，那么只有来自这些 CIDR 的流量才能访问到你的 LoadBalancer 服务，其他 IP 的流量会被直接丢弃。如果你不配置这个字段，那就意味着没有限制，任何外部 IP 都可以访问。

需要注意的是，这个检查只对外部流量生效，集群内部的 Pod 或者主机进程访问 LoadBalancer 服务是不受这个限制的，无论你是否配置了白名单。这个功能默认是开启的，但如果你的云提供商比如 AWS 已经实现了类似的功能，你可能想把它关掉。相反，如果你用的是 GKE 的内部 TCP/UDP 负载均衡器，它本身没有这个安全检查，那你就必须保持 Cilium 的这个功能开启，否则外部流量就没法被安全地限制了。

刚才我们提到了 LoadBalancerSourceRanges 的默认行为是只作用于 LoadBalancer 服务本身。这意味着，如果你的 LoadBalancer 服务同时创建了 NodePort 和 ClusterIP 服务，那么这个白名单只对 LoadBalancer 服务生效，NodePort 和 ClusterIP 服务不受影响。如果你希望把白名单 CIDR 应用到所有这些服务，有两种方法：一种是通过 service.cilium.io/type 注解，告诉 Cilium 只创建 LoadBalancer 服务，不创建 NodePort 和 ClusterIP。另一种是通过 Helm 安装时设置 bpf.lbSourceRangeAllTypes=true，这样 Cilium 就会把白名单 CIDR 应用到所有类型的外部服务。另外，LoadBalancerSourceRanges 默认是允许列表，也就是说，只有白名单里的 IP 才能访问。但如果你希望反过来，禁止白名单里的 IP 访问，允许其他所有 IP 访问，你可以通过 service.cilium.io/src-ranges-policy 注解，把值设为 deny。这样就变成了一个拒绝列表。Kubernetes 的服务管理中，有一个叫做 service.kubernetes.io/service-proxy-name 的注解。这个注解允许你指定哪些服务应该由 Cilium 或者 kube-proxy 来管理。Cilium 尊重这个注解。

默认情况下，Cilium 的服务代理名称是空字符串，这意味着它只管理那些没有 service.kubernetes.io/service-proxy-name 标签的服务。如果你想让 Cilium 只管理带有特定标签的服务，比如只管理那些服务代理名称为 cilium 的服务，你可以通过配置 k8s.serviceProxyName 选项来设置。这个功能对于精细化管理服务代理的范围非常有用，可以避免 Cilium 无意中管理了某些你不希望它管理的默认服务。

## Traffic Distribution and Topology Aware Hints

在大型集群中，如何高效地分配流量，避免跨地域访问，是一个重要的优化点。Kubernetes 提供了 Topology Aware Routing 和 Traffic Distribution 这两个特性。Cilium 的 eBPF 替代方案完美地支持了这两个功能。它们的核心机制是通过在 EndpointSlices 上设置 hints，也就是提示信息，告诉 Cilium 的负载均衡器，哪些服务端点位于同一个拓扑区域内，比如同一个机房或者同一个可用区。这样，当 Cilium 需要选择一个后端服务时，它会优先选择那些位于同一个区域的端点，从而减少网络延迟，提高性能。

要启用这个功能，只需要设置 loadBalancer.serviceTopology=true 就行了。



服务负载均衡需要知道每个后端节点的 L2 MAC 地址，才能进行二层转发。但是，我们不能在 BPF 的快速路径上动态地去解析邻居，因为那会引入延迟。所以，Cilium 在后台默默地进行邻居发现。

- 早期版本 1.10 及以下，Cilium Agent 自带了一个 ARP 解析库，负责发现邻居并把它们作为永久条目推送到内核。
- 但从 1.11 开始，Cilium 改变了策略，它不再自己做 ARP，而是完全依赖 Linux 内核来做邻居发现。无论是 IPv4 还是 IPv6，Cilium 都支持。
- 对于内核 5.16 或更高版本，Cilium 会检测到内核支持 managed 邻居条目，并且会把新加入的节点的 L3 地址推送给内核，标记为 managed extern_learn，这样内核就能自动管理这些邻居条目的刷新和可达性。
- 对于内核 5.15 或更低版本，Cilium 也会推送 L3 地址，但会通过一个周期性刷新机制来保持邻居条目的活跃状态。你可以通过 --enable-l2-neighbor-discovery=false 来禁用邻居发现，但那样可能会导致某些包丢失，尤其是在中间节点转发时。

Cilium 的邻居发现也支持多设备环境，比如一个节点有多个网卡连接到不同的网络，Cilium 会为所有可能的下一跳设备都去发现邻居。默认情况下，Kubernetes 的 ClusterIP 服务只能被集群内部的 Pod 访问，外部流量是无法直接访问的。Cilium 的 eBPF 替代方案也遵循这个原则。但是，如果你确实需要从集群外部访问某个 ClusterIP 服务，比如用于某些特定的测试或者调试场景，你可以通过设置 bpf.lbExternalClusterIP=true 来允许这种访问。需要注意的是，这通常不是一个推荐的生产环境配置，因为它打破了服务的隔离性，可能会引入安全风险。所以，只有在非常明确的需求下，才应该考虑开启这个选项。

## Observability

网络出了问题，我们怎么知道发生了什么？可观测性是关键。Cilium 提供了 Hubble 和 cilium monitor 这两个强大的工具来帮助我们追踪网络事件。特别是对于 socket-based 的负载均衡，也就是 socket LB，我们可以通过 Hubble 来观察数据包在经过 Cilium 转换前后的状态变化。比如，你可以看到一个请求从客户端 Pod 发送到服务端口，然后经过 Cilium 的转换，最终被路由到哪个后端 Pod。Hubble 的输出会清晰地显示这些转换前后的事件。

如果你的 cilium Agent 无法正确地检测到 Pod 的 cgroup 路径，Hubble 可能会报错，这时候，你可以退而求其次，使用 cilium-dbg monitor 命令来直接追踪数据包，虽然信息量会少一些，但也能提供关键的调试线索。排查问题时，有时候会遇到集群IP服务无法访问的情况。这时候，一个常见的检查点是 BPF cgroup 程序是否正确地附加到了主机的 cgroup 根目录上。默认情况下，Cilium 的 cgroup 根目录是 /run/cilium/cgrouppv2。你需要检查 bpftool cgroup tree /run/cilium/cgrouppv2/ 的输出，看看有没有 connect、sendmsg、recvmsg 等相关的 BPF 程序被成功附加。

特别要注意的是，如果你的容器运行时比如 containerd 在使用 cgroup namespace 模式，Cilium Agent 可能会错误地把 BPF 程序附加到虚拟化的 cgroup 根目录下，这样就无法正确地拦截和处理 Pod 的流量，导致负载均衡失效。确保你的容器运行时在 cgroup namespace 模式下运行，这是个关键的配置点。任何技术都有可能遇到已知的问题。







```
kubectl -n kube-system exec ds/cilium -- cilium-dbg status | grep \ kubeProxyReplacement
```





## 





## Neighbor Discovery