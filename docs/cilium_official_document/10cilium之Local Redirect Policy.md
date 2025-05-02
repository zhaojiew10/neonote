上节课我们深入探讨了端点管理与容器技术集成的方方面面，现在，我们将目光聚焦于一个更具体、更强大的工具——Cilium 的 Local Redirect Policy，简称 LRP。

## Local Redirect Policy

这不仅仅是简单的流量转发，它利用了 eBPF 的强大能力，为我们带来了前所未有的网络控制灵活性。在深入细节之前，我们先确保基础环境就绪。就像任何精密仪器需要稳定的电源一样，Cilium 的核心组件——Agent 和 Operator，必须正常运行。

```
$ kubectl -n kube-system get pods -l k8s-app=cilium
$ kubectl -n kube-system get pods -l name=cilium-operator
```

通过这两个命令，我们快速检查 kube-system 命名空间下的 cilium 和 cilium-operator Pod 是否都处于 Running 状态，并且没有重启记录。这一步是确保后续所有操作的基础，是整个演示的基石。

确认了 Agent 和 Operator 运行正常后，我们还需要验证一个关键的 Kubernetes 对象——Custom Resource Definition，也就是 CRD。

```
$ kubectl get crds
NAME                                     CREATED_AT
ciliumlocalredirectpolicies.cilium.io    2020-08-24T05:31:47Z
```

这个名为 ciliumlocalredirectpolicies.cilium.io 的 CRD 就是我们用来配置 LRP 的蓝图。它的存在，意味着 Kubernetes 已经认识并准备好处理这种特殊的 Cilium 资源。

现在，我们来谈谈 LRP 的核心机制：负载均衡。Cilium 提供了两种实现方式：**Socket-level 和 TC Load Balancer**。你可以把它们想象成两种不同的交通指挥系统。**Socket-level 更像是直接在应用层面进行路由，而 TC 则更底层，基于 Linux 的网络队列**。选择哪种模式，完全取决于你的业务场景和对服务处理方式的偏好。没有绝对的好坏，只有最适合你的方案。

### 完全替换模式

第一种配置模式，我们称之为完全替换模式。如果你的目标是彻底告别传统的 kube-proxy，拥抱 Cilium 的 eBPF 实现带来的性能优势，那么这个模式就是为你量身定制的。它不仅接管了服务发现，还接管了负载均衡，让你的网络栈更加纯粹、高效。

这种配置方式可以充分利用 Cilium 的 eBPF 实现，获得最佳性能和灵活性，同时简化网络配置。

```yaml
kubeProxyReplacement: true 
LocalRedirectPolicy: true
```

### 主机命名空间隔离

第二种模式，我们称之为“主机命名空间隔离”模式。它在完全替换的基础上，增加了一个关键参数：socketLB hostNamespaceOnly true。

```yaml
kubeProxyReplacement: true
socketLB:
	hostNamespaceOnly: true
localRedirectPolicy: true
```

这个配置的作用是什么？它允许你在 Pod 的命名空间里，比如你的**应用 Pod 里，部署一些特殊的、自定义的重定向规则，而这些规则不会干扰到 Cilium 主机层面的 Socket-level 负载均衡器**。这就像在主干道旁边，允许你开辟一条小路，专门用于处理特定的交通，互不干扰。这对于那些需要在 Pod 内部实现复杂流量控制逻辑的场景非常有用。

### 仅启用 Socket-level 负载均衡器

第三种模式，我们称之为“仅启用 Socket-level 负载均衡器”模式。这种模式下，你仍然选择保留 kube-proxy 来负责整个集群的服务发现和负载均衡，但同时，你又希望借助 Cilium 的 Local Redirect Policy 来实现某些特定的流量重定向。这就像你保留了传统的交通指挥系统，但同时引入了 Cilium 的 eBPF 技术，专门用来处理那些需要快速、高效的本地流量转发场景。

kubeProxyReplacement 保持为 false，但启用 socketLB 和 localRedirectPolicy。这是一种混合模式，既利用了现有生态，又引入了新技术。

```
kubeProxyReplacement: false
socketLB:
	enabled: true
localRedirectPolicy: true
```

### 仅处理 ClusterIP 服务

第四种模式，也是最保守的一种，我们称之为“仅处理 ClusterIP 服务”模式。在这种模式下，你完全依赖 kube-proxy 来处理所有服务相关的流量，但你希望 Cilium 的 LRP 能够介入，专门处理来自 Pod 命名空间内的 Pod 对 ClusterIP 服务的访问。这就像你让传统的交通警察负责大部分路口，但特别指定一个区域，由 Cilium 的 eBPF 路由器来处理特定的本地交通。

注意，这里的 Pod 流量指的是 Pod 之间的通信，而不是 Pod 访问主机网络的流量。kubeProxyReplacement 为 false，仅启用 localRedirectPolicy。这种模式适合那些只想在 Pod 层面实现一些本地优化，但不想大规模替换现有网络架构的场景。

理论讲完了，我们开始动手搭建实验环境。首先，我们需要准备两个 Pod：一个后端 Pod，作为流量的最终目的地；另一个客户端 Pod，用来发起请求，触发流量重定向。这个后端 Pod 的配置非常关键，它的标签、端口和协议必须与后续我们将创建的 LRP 规则中指定的匹配项完全一致。这里我们创建了一个名为 lrp-pod 的 Pod，它运行 Nginx，监听 80 端口的 TCP 流量，并且打上了 app等于proxy 的标签。这个标签是后续选择后端 Pod 的关键依据。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: lrp-pod
  labels:
    app: proxy
spec:
  containers:
    - name: lrp-pod
      image: nginx
      ports:
        - containerPort: 80
          name: tcp
          protocol: TCP
```

部署完后端 Pod 之后，我们立刻检查它的状态。通过 kubectl get pods 并 grep lrp-pod，我们可以快速确认这个 Pod 是否成功运行，状态是否为 Running。这一步非常重要，确保我们的目标服务器已经准备好接收被重定向的流量。

## 部署和测试

现在，我们部署客户端 Pod。这个 Pod 的作用是产生我们想要测试的流量。我们这里使用了一个现成的示例，mediabot，它会尝试访问一个特定的域名。部署它，然后等待它就绪。这个客户端 Pod 将成为我们验证 LRP 效果的探针。

```
$ kubectl create -f https://raw.githubusercontent.com/cilium/cilium/1.17.3/examples/kubernetes-dns/dns-sw-app.yaml
$ kubectl wait pod/mediabot --for=condition=Ready
$ kubectl get pods
NAME                             READY   STATUS    RESTARTS   AGE
pod/mediabot                     1/1     Running   0          14s
```

环境搭建完成，现在我们进入核心环节：创建 CiliumLocalRedirectPolicy 自定义资源。这是配置 LRP 的核心步骤。

Cilium 提供了两种主要的匹配方式来定义需要重定向的流量。

- 第一种是 AddressMatcher，它基于 IP 地址和端口协议来匹配流量。
- 第二种是 ServiceMatcher，它基于 Kubernetes Service 的名称和命名空间来匹配。

### AddressMatcher

选择哪种方式，取决于你希望重定向流量的来源和目标。接下来，我们将分别详细讲解这两种配置方式。我们先来看 AddressMatcher。

```yaml
apiVersion: "cilium.io/v2"
kind: CiliumLocalRedirectPolicy
metadata:
  name: "lrp-addr"
spec:
  redirectFrontend:
    addressMatcher:
      ip: "169.254.169.254"
      toPorts:
        - port: "8080"
          protocol: TCP
  redirectBackend:
    localEndpointSelector:
      matchLabels:
        app: proxy
    toPorts:
      - port: "80"
        protocol: TCP
```

这种方式非常直观，你告诉 Cilium所有发往 IP 地址 169.254.169.254，端口 8080 的 TCP 流量，都要被重定向到哪里去。这里，我们通过 redirectBackend 指定了后端 Pod 的选择器：标签为 app等于proxy 的 Pod。并且，我们还指定了后端 Pod 的端口是 80。注意，这里的 toPorts 可以命名，比如你可以把前端端口 8080 命名为 web，然后在后端端口里也用 web 对应 80，这样就实现了端口映射。这种配置方式适用于需要精确控制 IP 地址和端口的场景。这就是一个具体的 AddressMatcher 配置示例。我们定义了一个名为 lrp-addr 的 LRP。在 redirectFrontend 里，我们明确指定了 IP 地址 169.254.169.254 和端口 8080。在 redirectBackend 里，我们使用了 localEndpointSelector 匹配了标签为 app等于proxy 的 Pod，并且指定了后端端口是 80。这个配置文件清晰地表达了我们的意图：将特定 IP 和端口的流量，重定向到带有特定标签的 Pod 的特定端口。配置好 YAML 文件后，我们使用 kubectl apply -f 命令将其应用到集群中。

然后，我们立刻用 kubectl get ciliumlocalredirectpolicies 命令来检查这个新创建的 LRP 是否真的存在。如果能看到 lrp-addr，并且有 AGE 信息，那就说明配置成功。

### ServiceMatcher

接下来，我们看第二种匹配方式：ServiceMatcher。这种方式更符合 Kubernetes 的原生范式，它直接匹配 Kubernetes Service 的名称和命名空间。

比如，我们想把所有访问名为 my-service 服务的流量，都重定向到某个地方。使用 ServiceMatcher，你需要指定服务名和命名空间。同样，你也可以指定要重定向的服务端口，甚至可以指定只重定向特定的服务端口。这在需要基于服务进行流量控制时非常方便。这是 ServiceMatcher 的一个示例。

```yaml
apiVersion: "cilium.io/v2"
kind: CiliumLocalRedirectPolicy
metadata:
  name: "lrp-svc"
spec:
  redirectFrontend:
    serviceMatcher:
      serviceName: my-service
      namespace: default
  redirectBackend:
    localEndpointSelector:
      matchLabels:
        app: proxy
    toPorts:
      - port: "80"
        protocol: TCP
```

我们创建了一个名为 lrp-svc 的 LRP。在 redirectFrontend 中，我们使用了 serviceMatcher，指定了服务名为 my-service，命名空间为 default。在 redirectBackend 中，我们再次使用了 localEndpointSelector 匹配了标签为 app等于proxy 的 Pod，并且指定了后端端口是 80。这个例子展示了如何基于 Service 对象进行流量重定向，逻辑上更贴近 K8s 的服务发现机制。

同样，我们使用 kubectl apply -f 命令应用这个 ServiceMatcher 配置。然后，再次使用 kubectl get ciliumlocalredirectpolicies 来确认 lrp-svc 已经成功创建。这个过程和 AddressMatcher 的应用是一致的，只是配置的逻辑有所不同。

配置完成后，我们就可以开始验证这两种方式的效果了。在使用 LRP 时，我们需要了解几个关键的限制和注意事项。

- 首先，**LRP 只对新建立的连接生效，对于已经存在的连接，可能不会立即重定向**。如果你的客户端 Pod 之前已经连接到某个服务，那么即使你设置了 LRP，这些连接可能仍然会走原来的路径。为了确保所有连接都按照新规则走，通常需要重启客户端 Pod。
- 其次，目前 Cilium 的 LRP 不支持动态更新。如果你想修改策略，比如修改重定向的目标 Pod 或端口，你需要先删除旧的策略，然后重新创建一个新的。这在一定程度上增加了管理的复杂度，但这也是当前版本的现实。

## 实际场景中的应用

理论和实践都讲完了，现在我们来看看 LRP 在实际场景中的应用。

- 第一个经典案例是 Node-local DNS Cache。在大型集群中，DNS 查询是应用启动和运行时的常态。如果每次查询都要经过 kube-dns 服务，再经过网络，效率会比较低。通过 LRP，我们可以将 Pod 的 DNS 请求，比如对 10.0.0.10:53 的请求，重定向到运行在同一个 Node 上的 Node-local DNS Cache Pod。这样，Pod 就可以直接访问本地的缓存，大大减少了跨节点的网络延迟，显著提升了 DNS 解析速度，尤其是在集群规模较大的情况下，效果非常显著。

- 第二个应用案例是 EKS 上的 kiam redirect。在 Amazon EKS 集群中，Pods 需要访问 AWS Metadata Server 来获取安全凭证。直接访问存在安全风险，而且可能被滥用。kiam 是一个专门的工具，用于拦截和控制这些请求。我们可以利用 Cilium 的 LRP，将所有 Pod 发往 AWS Metadata Server 的 IP 地址（通常是 169.254.169.254）的流量，重定向到运行在同一个 Node 上的 kiam agent。这样，kiam 就可以安全地检查和转发这些请求，防止恶意访问。这极大地提升了 EKS 集群的安全性。
- 对于那些追求极致性能和复杂场景的用户，Cilium 还提供了更高级的配置选项。比如，skipRedirectFromBackend。这个参数允许你配置一个行为：当流量从后端 Pod 返回到前端时，如果目标仍然是前端，那么 Cilium 就不再进行重定向，而是直接将流量转发到最初的目的地。这在某些需要绕过重定向循环的场景下非常有用。不过，这个高级功能需要 Linux Kernel 版本至少 5.12 以上，因为依赖于 SO_NETNS_COOKIE 的 getsockopt 功能。

总结一下，Cilium 的 Local Redirect Policy 是一个非常强大的工具。它利用 eBPF 的高性能，提供了低延迟、高吞吐量的流量控制能力。它提供了灵活的配置模式，可以适应不同的网络架构和需求。无论是加速 DNS 查询，还是增强安全访问控制，LRP 都能提供有效的解决方案。它极大地扩展了 Cilium 的应用场景，让网络管理变得更加智能和高效。