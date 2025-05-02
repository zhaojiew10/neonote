现在，我们将目光转向网络数据平面，特别是Cilium是如何利用eBPF等先进技术来构建高效、可扩展的网络策略执行机制的。这不仅仅是理论，更是现代云原生环境中不可或缺的实践。Cilium的核心在于拥抱现代内核技术，尤其是eBPF和XDP。

## eBPF

扩展伯克利包过滤器，它提供了一种安全、高效的方式，让我们的应用程序能够直接访问内核数据，而无需像传统方式那样加载内核模块，这大大降低了风险和复杂度。而XDP，即表达式数据路径，更是快到极致，它直接在内核网络栈的入口处，也就是网络接口卡的驱动程序之前，就对数据包进行处理。这就像在高速公路上的收费站，直接拦截，效率极高。

当然，Cilium也巧妙地结合了传统的TC钩子和Socket钩子，提供了更灵活的策略和连接控制能力。这种组合拳，让Cilium在性能和功能之间取得了很好的平衡。有了这些底层技术，Cilium构建了一系列精巧的网络组件，就像乐高积木一样，可以灵活组合。

- 首先是Prefilter，这个XDP驱动的预过滤器，它就像一个高效的门卫，利用CIDR映射快速判断哪些流量是无效的，比如目的地不是有效端点，直接丢弃，避免后续处理的负担。
- 接着是Endpoint Policy，这是Cilium的核心，负责根据端点的身份和策略，执行L3、L4级别的规则，决定流量去向。
- 还有Service，负责服务负载均衡，无论是集群内部还是外部访问，都能智能路由。
- L3 Encryption则提供了IPsec加密，保障数据安全。
- 特别值得一提的是Socket Layer Enforcement，它利用Socket操作和收发钩子，一旦TCP连接建立，后续数据包就能走快速通道，避免重复策略检查，性能提升非常显著。
- 最后是L7 Policy，通过Envoy这样的用户空间代理，实现更精细的应用层控制，比如HTTP路由。

现在我们来看一个具体的场景：同一个节点上的两个Pod之间如何通信。数据包从一个Pod发出，经过TC Endpoint的钩子，进入Cilium的数据平面

- 首先，Prefilter会快速检查，如果目标地址无效，比如不是有效的Pod IP，就直接丢弃。
- 如果通过，就进入Endpoint Policy进行检查。这个组件会根据源和目标Pod的标识信息，查找对应的策略，决定是允许通过、转发到本地另一个端点、还是转发到Service模块。
- 如果启用了L7 Policy，还会进行应用层的检查，比如HTTP请求是否符合规则。
- 当使用Socket Layer Enforcement加速路径时，一旦TCP连接建立，后续的握手和数据包交换，就可以通过Socket Hook直接在内核层面完成，绕过了中间的Policy检查，速度非常快，这对于高频的内部通信来说，至关重要。

接下来是Pod访问外部网络的场景。这个过程稍微复杂一些。数据包从Pod出发，经过TC Endpoint，进入Cilium。

- Prefilter和Endpoint Policy会进行过滤和策略检查。
- 如果需要，L7 Policy也会介入。
- 如果启用了L3 Encryption，数据包会被加密。
- 然后，如果启用了Overlay网络，比如VXLAN，数据包会被封装起来，通过Overlay接口cilium_vxlan发送出去，最终到达外部网络。
- 同样，如果使用了Socket Layer Enforcement加速，TCP连接建立后，后续的数据包也能走快速通道，减少不必要的策略检查和处理开销。

这个流程确保了Pod对外部网络的访问是受控且安全的。现在我们来看外部网络如何访问集群内的Pod。这个流程是入口到端点。

- 数据包从网络接口卡进入，经过TC  NIC的钩子。
- 如果是加密的IPsec包，L3 Encryption会先进行解密。
- 然后是Prefilter，快速过滤掉无效流量。
- 如果需要，L7 Policy会进行应用层检查。
- 如果目标是服务，Service模块会介入，将流量分发给后端的Pod。
- 然后，Endpoint Policy会根据源和目标的策略，决定是否允许访问，并将数据包转发到目标Pod。
- 如果启用了Overlay，VXLAN或Geneve会负责解封装数据包。
- 同样，Socket Layer Enforcement加速路径在TCP连接建立后也能发挥作用。这个过程确保了外部流量只能访问到符合策略的Pod。

## eBPF Map

刚才我们提到了Prefilter、Endpoint Policy、Service等等这些组件，它们是如何高效地进行策略匹配和状态管理的呢？答案就是eBPF Map。你可以把eBPF Map想象成一个高速缓存，专门用来存储和查找数据，供eBPF程序使用。

它就像一个共享数据的容器，连接着eBPF程序和内核。有了这些Map，Cilium才能在极短的时间内，根据数据包的源地址、目的地址、端口、协议等信息，快速找到对应的策略或者状态信息，比如连接跟踪状态。常见的Map类型有Hash、Array、Stack、Queue等等。

但是，Map不是无限大的，它有容量限制。这个容量限制直接关系到Cilium的性能和扩展性，配置不当可能会导致性能瓶颈。这张表展示了Cilium中各种关键eBPF Map的默认容量限制。可以看到，像Connection Tracking，也就是连接跟踪，它区分TCP和UDP，TCP的容量是1M，UDP是256k。NAT Map，邻居表，端点Map，服务负载均衡Map，策略Map，都有各自的默认大小。这些数字代表了单个节点上能够容纳的并发连接数、端点数量、策略条目数量等等。比如，Connection Tracking的1M条目，意味着理论上可以支持100万个并发连接。这些默认值是经过权衡考虑的，但实际应用中，我们需要根据集群的规模和负载，仔细评估这些Map是否足够大，或者是否需要进行调整。

默认值可能不够用，怎么办？Cilium提供了两种方式来调整Map的容量。

- 第一种是通过命令行参数，比如--bpf-ct-global-tcp-max，你可以手动指定TCP连接跟踪表的最大大小。
- 第二种是更智能的方式，使用--bpf-map-dynamic-size-ratio。这个参数允许你指定一个比例，比如0.0025，意味着Cilium会根据节点的总内存大小，动态计算出一个合适的Map容量，通常是占用总内存的0.25%。

这种方式特别适合大型集群，可以根据节点的资源情况自动调整，避免了手动配置的麻烦。这张表格对比了Cilium和kube-proxy在CT表大小上的差异，Cilium基于内存大小，理论上更灵活，可以根据节点资源动态调整。当然，动态调整也会影响性能，需要根据实际需求权衡。

我们再聚焦一下Service负载均衡相关的Map，也就是cilium_lb4_6_services_v2。这个Map专门用来存储服务负载均衡的配置信息，比如ClusterIP和NodePort服务。它的默认大小是64k。如果这个Map满了，Cilium可能就无法正确处理Service的更新，或者无法创建新的Service。那么，如何估算这个Map的大小呢？一个简单的公式是：每个服务占用的Map条目数等于服务关联的Pod数量乘以服务定义的端口数量。总的Map条目数就是所有服务占用的条目数之和。

```
LB Map Entries per Service = (Pods per Service) * (Ports/Protocols per Service)
LB Map Entries = (Number of LB Services) * (Avg Pods per Service) * (Avg Ports/Protocols per Service)
```

比如，如果你有100个服务，每个服务平均关联5个Pod，每个服务有1个端口，那么总的Map条目数就是100乘以5乘以1等于500。

需要注意的是，一旦Cilium节点上的Map创建完成，如果想重新调整大小，重启Cilium可能会导致服务连接中断，所以最好在安装Cilium之前就仔细规划好Map的大小。Cilium虽然强大，但并非完全抛弃传统。它巧妙地利用了Iptables。Iptables是Linux内核自带的防火墙，历史悠久，功能非常强大，社区支持广泛。在某些情况下，比如内核版本不支持某些eBPF特性，或者需要实现一些特定的、eBPF尚未完全覆盖的功能时，Cilium会退一步，使用Iptables作为后盾。这就像一个现代的超级英雄，虽然拥有强大的超能力，但有时也会借助一些传统武器。这种兼容性设计，保证了Cilium在各种复杂环境下的功能完备性，也降低了迁移和部署的风险。

特别是对于Kube-proxy，Cilium需要与之协同工作，共享Iptables规则，确保服务发现和网络策略的正确性。这张图展示了Cilium和Kube-proxy是如何在Iptables层面进行集成的。Kube-proxy是Kubernetes的默认服务代理，它依赖于Iptables来实现服务发现和负载均衡。

而Cilium，虽然有自己的eBPF数据平面，但在某些场景下，比如处理Service的流量，或者执行一些Network Policy时，它会插入Iptables规则。这种集成方式的好处是，可以利用现有的Kube-proxy基础设施，实现平滑过渡，降低迁移成本。用户可以逐步将网络策略从Iptables迁移到Cilium的eBPF数据平面，享受更高的性能。当然，随着eBPF技术的成熟，未来趋势很可能是Cilium的数据平面完全接管网络策略，逐步减少对Iptables的依赖。



