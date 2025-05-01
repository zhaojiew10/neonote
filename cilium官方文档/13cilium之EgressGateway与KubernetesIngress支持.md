https://docs.cilium.io/en/stable/network/egress-gateway/egress-gateway/

## Egress Gateway

Cilium Egress Gateway它就像一个智能的交通指挥中心，专门处理从集群内部发往特定外部网络的流量。**核心机制是SNAT，也就是源地址转换，它能将出站数据包的源地址伪装成我们预设的、可预测的网关节点IP地址**。这在需要与传统防火墙或特定网络环境交互时，简直是量身定做。

为什么我们需要Egress Gateway呢？想象一下，你管理着一个复杂的网络环境，里面有各种老旧的防火墙和安全策略，它们通常只认IP地址。但Kubernetes的Pod和节点的IP地址是动态变化的，今天是这个，明天可能就变了。传统的SNAT虽然能解决一部分问题，但节点IP也可能变，这就像给一个经常搬家的人配钥匙，太麻烦了！

Egress Gateway就是来解决这个问题的。它提供一个固定、可预测的IP地址，让网络策略配置变得简单明了，而且可以更精细地控制哪些Pod可以访问外部网络，大大提升了安全性。要理解Egress Gateway，我们得认识几个关键角色。

- 首先是网关节点，它们就是被指定的、负责处理出站流量的特殊节点。
- 然后是出站策略，这是我们的规则手册，告诉Cilium哪些Pod的流量需要通过哪个网关节点，并且如何进行SNAT。

Cilium实现这一切的核心技术是BPF Masquerading，这是一种高效且直接在内核层面操作的SNAT方式。同时，为了达到最佳性能，Egress Gateway通常依赖于kube-proxy的替代方案，而不是传统的kube-proxy。启用Egress Gateway其实很简单，主要就是通过Helm修改配置。配置好后，别忘了重启Cilium Agent和Operator，让这些新配置生效。

```
$ helm upgrade cilium cilium/cilium --version 1.17.3 \
   --namespace kube-system \
   --reuse-values \
   --set egressGateway.enabled=true \
   --set bpf.masquerade=true \
   --set kubeProxyReplacement=true
```

这个过程通常很快，但确保所有节点都已更新。配置好基础后，我们来谈谈如何定义具体的策略。Cilium使用一个名为CiliumEgressGatewayPolicy的自定义资源定义来管理这些策略。

注意，这个资源是集群范围的，所以你不需要指定命名空间。最基础的就是给它起个名字，比如example-policy。这只是个开始，真正的魔法在于spec部分。策略的核心在于选择哪些Pod的流量需要被路由和SNAT。我们使用selectors字段来实现。最常用的是podSelector，你可以通过matchLabels或matchExpressions来精确匹配Pod标签。比如，你可以选择所有带有org等于empire和class等于mediabot标签的Pod。

更进一步，如果你想让策略只在某个特定节点上生效，可以添加nodeSelector，比如只针对node1上的Pod。选定了源Pod之后，我们还需要指定这些流量应该去往哪里。destinationCIDRs字段允许我们定义一个或多个IPv4目标网络范围。比如，你可以把所有出站流量都指向0.0.0.0斜杠0，也就是整个互联网。但有时候，我们可能只想路由一部分流量，比如只路由到某个特定的子网。

更灵活的是，我们可以使用excludedCIDRs来排除特定的CIDR。比如，如果destinationCIDRs是0.0.0.0斜杠0，但你不想把流量路由到192.168.0.0斜杠24，就可以在excludedCIDRs中添加它。另外，Cilium会自动排除掉内部集群的IP，避免不必要的SNAT。

现在，我们来指定哪个节点作为网关，并且使用哪个IP地址进行SNAT。这通过egressGateway字段来完成。同样，我们使用nodeSelector来选择网关节点，比如选择带有特定标签的节点。然后，我们需要指定SNAT使用的IP地址。这里有三种方式：

- 要么明确指定一个egressIP，比如10.168.60.100；
- 要么指定一个接口interface，Cilium会自动使用该接口的第一个IPv4地址；
- 或者，你也可以什么都不指定，Cilium会尝试使用默认路由的接口。

记住，egressIP和interface不能同时使用，否则策略会被忽略。让我们把前面讲的都组合起来，看一个完整的例子。

```yaml
apiVersion: cilium.io/v2
kind: CiliumEgressGatewayPolicy
metadata:
  name: egress-sample
spec:
  selectors:
  - podSelector:
      matchLabels:
        org: empire
        class: mediabot
        io.kubernetes.pod.namespace: default
    nodeSelector:
      matchLabels:
        node.kubernetes.io/name: node1
  destinationCIDRs:
  - "0.0.0.0/0"
  egressGateway:
    nodeSelector:
      matchLabels:
        node.kubernetes.io/name: node2
    egressIP: 10.168.60.100
```

这个名为egress-sample的策略会将所有带有org=empire和class=mediabot标签、位于default命名空间的Pod（如果在node1上），并且它们的流量目的地是0.0.0.0/0（即整个互联网）的请求，通过带有node.kubernetes.io/name等于node2标签的节点，使用10.168.60.100这个IP地址进行SNAT。是不是很清晰？

这就是Egress Gateway策略的力量所在。理论知识有了，怎么验证它是否真的工作呢？我们来模拟一个场景。假设我们有一个运行在node1上的Pod，它需要访问一个外部的Nginx服务。首先，我们部署这个客户端Pod。然后，我们应用我们刚才定义的Egress Gateway策略。最后，我们去检查那个Nginx服务的访问日志，看看请求是从哪个IP地址来的。如果没有应用策略，我们期望看到的是node1的IP。如果策略生效了，我们期望看到的应该是我们指定的网关节点的IP地址，也就是10.168.60.100。这就是验证的关键步骤。

万一策略没生效，或者效果不对，怎么办？别慌，我们有工具来排查。你可以使用cilium-dbg bpf egress list命令来查看Cilium Agent上配置的Egress规则。这个列表会告诉你哪些源IP、哪些目标CIDR、哪些网关IP和Egress IP是匹配的。仔细核对，看看你的策略配置是否正确，特别是Pod和网关节点是否被打上了正确的标签。如果列表里没有你预期的条目，那多半就是标签问题了。

在使用Egress Gateway时，有几个地方需要注意。

- 新启动的Pod可能需要一点时间才能应用到策略，这意味着新Pod的初始流量可能不会经过网关。
- 它和某些其他功能，比如identity allocation mode kvstore和Cluster Mesh，是不兼容的。
- 目前Egress Gateway只支持IPv4，IPv6流量是不支持的。
- 如果你的内核版本比较老，比如低于5.10，可能会遇到一些路由选择问题，推荐使用5.10或更高版本的内核。
