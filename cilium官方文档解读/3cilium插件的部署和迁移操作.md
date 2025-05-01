现在，让我们聚焦于一个关键的部署细节——NoExecute污点。这看似简单的一个配置，实际上在Cilium的部署和运维中扮演着至关重要的角色，尤其是在云原生环境中。它直接关系到Pod的调度、网络的接管以及潜在的业务中断风险。我们来看NoExecute这个污点。它就像一把双刃剑。

- 一方面，它能有效地防止应用Pod在Cilium Agent完全准备好之前就被调度到节点上，这保证了网络策略的正确应用和Pod的网络环境一致性。
- 但另一方面，一旦这个污点被意外地重新添加到节点上，比如在云平台的自动伸缩、升级或者某些运维操作中，所有运行在该节点上的Pod就会被驱逐出去，直到Cilium有机会清除这个污点。这可能会导致应用中断，用户体验受损。要理解NoExecute的风险，我们得先搞清楚Kubernetes节点Node本身。

在云环境下，我们常说的节点，它背后可能是一个虚拟机实例，这个实例可能会被云平台进行各种操作，比如打补丁、重置文件系统，甚至被替换为一个全新的实例。但关键在于，这个新实例可能继承了原来节点的名称。这意味着什么？如果原来的节点已经运行了某些Pod，而新的节点实例上线后，虽然节点资源层面的污点被重新添加了，但那些已经在运行的Pod可能仍然会继续运行，因为它们是基于旧节点资源创建的，这就导致了这些Pod可能在没有Cilium管理的情况下运行，成为潜在的孤儿Pod，网络策略无法生效。更棘手的是，有时候污点的重新出现并非我们预期的节点升级或重置。比如，某些云平台的自动化操作，或者系统内部的某些故障，都可能导致节点的污点被意外地添加回来。这就好比一个幽灵，悄无声息地出现在你的节点上，然后触发Pod的逐出。这正是使用NoExecute时需要特别警惕的地方。因为一旦发生这种情况，你可能会面临业务中断的风险。所以，我们需要根据具体的环境和云服务商的特性，仔细评估是否使用NoExecute，以及如何使用。

所以，我们面临一个核心的权衡：要么接受一部分Pod可能在没有Cilium管理的情况下运行，这可能会带来网络隔离失效、流量丢失等问题；要么就接受使用NoExecute可能导致的意外Pod逐出，进而导致应用中断。Cilium官方文档倾向于推荐NoExecute，因为它被认为是目前在云环境中部署Cilium时，对业务中断影响最小的模式。当然，这并不意味着绝对安全，而是说，相比其他选项，它在大多数情况下能提供一个更稳定的过渡。最终的目标是，找到一个平衡点，将业务中断的风险降到最低。

讲完了NoExecute，我们把目光转向Cilium的部署方式。除了快速安装，Helm安装是另一种非常主流的方式。相比快速安装，Helm提供了更精细的配置控制。你可以根据自己的环境，比如网络拓扑、性能要求等，手动选择最适合的数据路径和IP地址管理模式。这对于需要高度定制化、或者在复杂环境下的部署来说，是必不可少的。虽然步骤稍微多一些，但灵活性和可控性是Helm安装的显著优势。

## helm部署cilium

讲完了NoExecute，我们把目光转向Cilium的部署方式。除了快速安装，Helm安装是另一种非常主流的方式。相比快速安装，Helm提供了更精细的配置控制。你可以根据自己的环境，比如网络拓扑、性能要求等，手动选择最适合的数据路径和IP地址管理模式。这对于需要高度定制化、或者在复杂环境下的部署来说，是必不可少的。虽然步骤稍微多一些，但灵活性和可控性是Helm安装的显著优势。

使用Helm安装Cilium，首先要做的是初始化。

- 第一步，添加Cilium的Helm仓库，这个命令 helm repo add cilium https://helm.cilium.io/ 就是干这个的。然后，你需要选择合适的Helm Chart版本。

  ```
  helm repo add cilium https://helm.cilium.io/
  ```

- 接下来，根据你的需求，可以通过 helm install 命令的 --set 参数来配置一些关键参数，比如数据路径、IPAM模式等。或者，你可以把这些配置写到一个 values.yaml 文件里，然后通过 --values 参数引用，这样管理起来更清晰。这些准备工作是后续安装成功的基础。

  ```
  helm install cilium cilium/cilium --version 1.17.3 --set eni.enabled=true,ipam.mode=eni,egressMasqueradeInterfaces=eth+,routingMode=native
  ```

现在，我们来看一个具体的部署场景：AWS EKS。**在EKS上，Cilium的默认安装配置会使用AWS ENI模式。这种模式充分利用了AWS的弹性网络接口，性能非常高，特别适合那些对网络吞吐量要求高的应用**。如果你需要在EKS上部署Cilium，强烈建议优先考虑这个模式。当然，**如果你的现有网络是基于AWS VPC CNI的，也可以选择将Cilium链式部署在VPC CNI之上，但这需要额外的配置**。EKS的安装指南也支持单区域、多区域、多可用区等多种架构。

在EKS上安装Cilium，有几个关键的要求。

- 首先，对于EKS托管的节点组，必须正确配置污点。推荐使用 node.cilium.io/agent-not-ready 这个键，值为 true，效果是 NoExecute。这确保了只有当Cilium Agent准备好接管网络时，应用Pod才会被调度到这个节点上。你可以通过 eksctl 的 ClusterConfig 文件在创建集群时就配置好这个污点。如果你是后期添加节点，也可以手动给节点添加这个污点。这个步骤至关重要，它直接关系到你的Pod能否被Cilium正确管理。
- 不过，ENI模式在EKS上也有一个限制。目前，它只支持IPv4。如果你需要在EKS上使用IPv6，那么你就不能使用ENI模式，必须选择其他的网络模式，比如Overlay模式。Overlay模式虽然也能在EKS上运行，但会有一些额外的考虑，比如出站流量的SNAT以及API Server访问受限的问题。所以，如果你在EKS上需要IPv6，就需要提前规划好，选择合适的替代方案。
- 在EKS上，AWS默认会用VPC CNI插件来管理ENI。而Cilium也需要管理ENI。这就会导致冲突。如果两个插件同时管理同一个ENI，网络就会乱套。所以，我们需要做一件事，就是去patch掉VPC CNI的aws-node DaemonSet，告诉它：嘿，这个节点上的ENI归Cilium管了，你别管了。这个patch操作是通过添加一个特定的NodeSelector来实现的，io.cilium/aws-node-enabled=true，这样就能确保Cilium对ENI的专有控制。

现在我们来看具体的Helm安装命令。这里给出的是一个示例，版本是1.17.3。

```shell
helm install cilium cilium/cilium --version 1.17.3 \
  --namespace kube-system \
  --set eni.enabled=true \
  --set ipam.mode=eni \
  --set egressMasqueradeInterfaces=eth+ \
  --set routingMode=native
```

关键在于后面的几个 --set 参数。eni.enabled=true 明确告诉Cilium要启用ENI模式。ipam.mode=eni 指定IPAM模式也要用ENI。egressMasqueradeInterfaces=eth+ 指定了SNAT的出口接口，通常用 eth+ 代表所有以太网接口。routingMode=native 则选择了原生路由模式。这些参数组合起来，就定义了我们在EKS上使用ENI模式的Cilium配置。

当我们**设置了 eni.enabled=true 和 routingMode=native 后，Cilium会为每个Pod分配一个全局路由的AWS ENI IP地址**。这意味着你的Pod可以直接访问VPC内的其他资源，就像使用VPC CNI插件一样。这种模式的好处是网络性能好，但代价是它依赖于EC2 API的特定权限。你需要确保你的Kubernetes集群有足够的权限去调用EC2 API来创建和管理ENI。这是使用ENI模式的前提条件。

**除了ENI模式，Cilium在EKS上还可以使用Overlay模式**。这种方式下，Pod获得的IP地址不是全局路由的，而是属于一个Overlay网络。好处是，理论上可以让你在单个节点上运行更多的Pod，因为不再受ENI数量限制。但是，**Overlay模式也有它的局限性。比如，Pod访问集群外部资源时，出站流量会被Cilium SNAT到节点的VPC IP**。更重要的是，<span style="color:red">EKS的API Server默认无法路由到这个Overlay网络，这意味着你需要特别注意，任何依赖Webhook的组件，要么需要部署在宿主机网络，要么通过Service或Ingress暴露。</span>

要切换到Overlay模式，配置相对简单。你只需要在Helm安装命令里，去掉 eni.enabled=true、ipam.mode=eni 和 routingMode=native 这几个参数。这样，Helm就会默认使用Overlay模式。

另外，由于VPC CNI本身也会添加一些iptables规则，为了确保Cilium的Overlay网络能够正常工作，最好在安装Cilium之前，先清理掉VPC CNI可能添加的nat表规则，比如执行 iptables -t nat -F。

在配置SNAT接口时，我们通常会用 eth+。但需要注意的是，不同的Linux发行版，网卡的命名规则可能不一样。比如，Amazon Linux 2 通常使用 eth+，而更新的 Amazon Linux 2023 则倾向于使用 ens+。如果你的集群里混用了这两种操作系统，目前还不支持混合部署。所以，务必检查你的集群节点，确认接口命名规则是否一致，然后在配置里使用正确的通配符，比如 eth+ 或 ens+。

安装完Cilium之后，还有一个重要的步骤，就是处理那些在Cilium部署之前就已经运行的Pod，也就是所谓的未管理Pod。如果你的集群是干净地创建的，没有预先设置污点，那么这些Pod可能不会被Cilium自动管理。为了确保所有Pod都受到Cilium网络策略的保护，你需要手动重启这些Pod。这里提供了一个kubectl命令，它会筛选出所有未使用HostNetwork模式的Pod，然后逐个删除，让它们重新启动，从而被Cilium接管。

```
kubectl get pods ... | grep '{none}' | xargs -l 1 -r kubectl 
```

安装完成之后，怎么验证Cilium是不是真的成功运行了呢？官方推荐使用Cilium CLI。你可以通过官方文档提供的脚本安装这个CLI。安装好后，运行 cilium status --wait 命令，就能看到Cilium、Operator、Hubble等组件的状态，以及Pod的管理情况。这个命令是检查Cilium健康状态的常用工具。光看状态还不够，我们还需要验证网络连通性。Cilium CLI 提供了 connectivity test 命令，可以帮你检查集群内部的网络是否正常。它会运行一系列测试，最后给出一个报告，告诉你哪些测试成功，哪些失败。如果遇到 too many open files 错误，通常是由于Pod进程打开了太多文件描述符，导致资源限制不足。这时可以尝试增加主机上的 inotify 资源限制。

## 迁移cni插件

现在我们来聊聊一个更复杂的话题：如何将一个已经运行的集群，从现有的CNI插件，比如Flannel、Calico，迁移到Cilium。这是一个非常实际的场景，但也是一个挑战。因为迁移过程可能会中断业务，所以我们需要尽可能地平滑过渡。幸运的是，Cilium提供了支持集群迁移的功能，可以帮助我们做到这一点。

在深入迁移细节之前，我们先回顾一下CNI的工作原理。当Kubelet创建一个Pod的沙箱时，它会调用当前配置的CNI插件来为这个Pod分配网络资源，包括IP地址、创建网络接口、建立Overlay等。**在迁移过程中，我们通常会修改 /etc/cni/net.d/ 目录下的配置文件，指向新的CNI插件**。但是，问题来了，那些在修改配置之前就已经创建的Pod，它们的网络配置还是由旧的CNI插件管理的。所以，我们需要一个策略来处理这些旧的Pod。

传统的迁移方法，通常是把整个集群的节点重启一遍，或者逐个节点重启，这样确实能完成迁移，但代价是会中断服务。Cilium提供了一种更优雅的方式：**双Overlay迁移**。它的核心思想是，在迁移过程中，同时运行两个Overlay网络，一个是旧的CNI，一个是新安装的Cilium。通过Linux路由表，可以将两个网络的流量分开，但又允许它们互相通信。这样，即使节点正在迁移，Pod也能保持连通性，大大减少了业务中断。要想实现平滑的双Overlay迁移，有几个前提条件。

- 首先，你需要为Cilium分配一个全新的、独立的Cluster CIDR，也就是IP地址段，这个地址段不能和你现有的CNI网络冲突。强烈推荐使用 Cluster Pool IPAM 模式来管理这个新的CIDR。
- 其次，你需要确保Cilium的Overlay网络和旧的CNI网络在协议或端口上是不同的。比如，如果旧的网络是Flannel的VXLAN，你可以让Cilium用VXLAN但换个端口，或者用GENEVE。
- 最后，你的现有CNI必须是基于Linux路由栈的，比如Flannel、Calico或AWS-CNI。当然，双Overlay迁移也不是万能的。它有一些已知的局限性。比如，目前还没有经过充分测试的场景包括基于BGP路由、从IPv4迁移到IPv6、从链式Cilium迁移到独立Cilium，以及迁移过程中存在现有NetworkPolicy的情况。
- 特别需要注意的是，为了防止迁移过程中旧的Pod和新的Cilium网络策略冲突，Cilium会自动禁用策略执行。你需要在迁移完成后，手动重新启用。

再次强调，使用 Cluster Pool IPAM 是非常推荐的，因为它能最大程度地避免IP冲突。最后，也是最重要的一个警告：迁移是一个高度依赖于你现有集群配置的过程。每个集群的网络、Pod、Service配置都不一样，所以迁移方案很难一概而论。因此，强烈建议你在正式迁移生产环境之前，先在测试集群或者实验室环境里完整地演练一遍。这能帮你发现潜在的问题，并调整你的策略。不要跳过这个测试环节，否则可能会在生产环境中遇到意想不到的麻烦。

我们来概览一下整个分阶段迁移的流程。

- 第一步，先安装Cilium，但让它处于Secondary模式，也就是只建立Overlay，但不接管Pod的网络。
- 第二步，开始逐个节点迁移。你可以选择一个节点，比如Worker节点，然后进行一系列操作。
- 第三步，当所有节点都迁移到Cilium之后，就可以移除旧的CNI插件了。
- 第四步，可以考虑重新启动节点，让它们完全使用新的Cilium网络。为了方便大家练习，我们可以使用Kind来快速搭建一个测试集群。

Kind是一个轻量级的Kubernetes集群，非常适合在本地开发和测试。这里展示了如何使用kind-config.yaml来配置集群，禁用默认

```shell
$ cat <<EOF > kind-config.yaml
apiVersion: kind.x-k8s.io/v1alpha4
kind: Cluster
nodes:
- role: control-plane
- role: worker
- role: worker
networking:
  disableDefaultCNI: true
EOF
$ kind create cluster --config=kind-config.yaml
$ kubectl apply -n kube-system --server-side -f https://raw.githubusercontent.com/cilium/cilium/1.17.3/examples/misc/migration/install-reference-cni-plugins.yaml
$ kubectl apply --server-side -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
$ kubectl wait --for=condition=Ready nodes --all
```

的CNI，然后分别应用Flannel的配置文件和kube-flannel.yml，最后等待所有节点就绪。

这样你就有了一个可以用来测试迁移的环境。在正式迁移前，我们需要做一些关键的配置选择。

- 第一步，选择一个全新的、独立的Cluster CIDR，这个地址段必须和你现有的网络，比如Flannel的10.244.0.0/16，完全区分开来。比如，在Kind集群里，我们可以选择10.245.0.0/16。
- 第二步，选择一个与现有CNI不同的封装协议或端口。比如，如果Flannel用的是VXLAN，我们可以让Cilium也用VXLAN，但指定一个非默认的端口，比如8473。
- 接下来，我们需要创建一个专门用于迁移的 Helm Values 文件，比如 values-migration.yaml。这里面有很多关键配置。比如，operator.unmanagedPodWatcher.restart 设置为 false，表示迁移过程中不要自动重启旧的Pod，因为我们手动控制。cni.customConf 和 cni.uninstall 都设置为 false，表示不安装Cilium的CNI配置文件，也不在移除旧CNI时删除Cilium的配置。ipam.mode 用 cluster-pool，clusterPoolIPv4PodCIDRList 填写我们选择的新CIDR。policyEnforcementMode 设置为 never，禁用策略。bpf.hostLegacyRouting 设置为 true，允许Cilium和旧网络之间的路由。
- 有了配置文件，就可以运行 helm install 命令来安装Cilium了。注意，这里用的是我们刚刚创建的 values-migration.yaml 文件。
- 安装完成后，再次运行 cilium status --wait，你会看到 cluster Pods: 0/3 managed by Cilium，表示Cilium已经安装好了，但还没有管理任何Pod。这说明我们的Secondary模式安装成功了。
