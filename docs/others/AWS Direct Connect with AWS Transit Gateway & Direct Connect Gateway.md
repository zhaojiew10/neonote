https://www.napafrica.net/wp-content/uploads/2021/11/Teraco-Tech-Day-AWS-Direct-Connect-with-AWS-Transit-Gateway.pdf

## 如何连接到全球的AWS？

![image-20251202210511171](https://s2.loli.net/2025/12/02/cMXkHEoaTl8dx41.png)

- 第一种，也是最基础的，就是通过公共互联网。这种方式简单直接，但缺点也很明显：它是共享网络，性能波动大，没有服务级别协议SLA保障，安全性也相对较低，除非应用层做了加密。
- 第二种，是使用IPSEC VPN或SDWAN。这比直接走互联网要好一些，支持多隧道聚合达到高带宽，但要实现10Gbps以上的吞吐量，设计复杂度会显著增加，同样缺乏SLA。

- 第三种，AWS Direct Connect。它提供了一条私有、专用的线路，绕过了公共互联网，不仅性能稳定、可预测，还提供了高达100Gbps的带宽选项，并且有SLA保障。对于需要高可靠性、低延迟、高安全性的连接，特别是大型企业，Direct Connect是更优的选择。


## Direct Connect的核心概念

Direct Connect本质上是连接您本地数据中心或办公室与AWS全球网络的一个物理接口。这个接口位于AWS Direct Connect位置，也就是一个特定的物理设施。通过这个专用连接，您的数据可以绕过公共互联网，实现与Amazon VPC或其他AWS服务的私有通信。

![image-20251202210525184](https://s2.loli.net/2025/12/02/CDvbGH8PwVfE6rx.png)

要建立Direct Connect连接，涉及多个角色。

- 服务提供商，他们负责提供最后一公里的物理连接。
- 合作伙伴，他们是AWS认可的专业人士，拥有部署Direct Connect的专长，他们协助而非转售。
- 转售商，他们是获得授权可以转售Direct Connect服务的合作伙伴。
- 共址提供商（colocation provider），他们是拥有并运营Direct Connect位置的物理设施的业主。

<img src="https://s2.loli.net/2025/12/02/unwXpIACQyhNriZ.png" alt="image-20251202210705213" style="zoom:67%;" />

理解这些角色有助于明确责任划分和成本构成。

### Direct Connect的连接类型

Direct Connect提供了两种主要的连接类型。第一种是专用连接，这是完全为单个客户物理隔离的线路，速度选项包括1G、10G和即将推出的100Gbps。这里的交叉连接由客户、合作伙伴或转售商的AWS账户拥有，也是专用于该客户的。

第二种是托管连接，由AWS合作伙伴代表客户逻辑上配置。它的速度范围更广，从50Mbps到10Gbps不等。注意，这里的交叉连接由合作伙伴或转售商的AWS账户拥有，是逻辑分区并在多个客户间共享的。无论哪种方式，最后一公里通常由服务提供商、合作伙伴或转售商拥有和管理。总成本需要考虑AWS端口小时费、数据传输费、交叉连接费以及最后一公里的费用。

### 部署拓扑和高可用性

部署Direct Connect时，有几种常见的拓扑选择。第一种，也是最简单的，就是您的本地设备直接位于同一个DX位置内，通过交叉连接接入。第二种，是在您的数据中心和DX位置之间建立一个电路，比如光纤专线。第三种，是通过服务提供商的网络，例如MPLS网络，延伸到DX位置。选择哪种方式取决于您的地理位置、现有网络基础设施以及预算要求。

业务连续性是任何关键连接设计的基石。使用Direct Connect时，如何规划？首先，强烈建议至少部署两条专用连接，并采用活动到活动模式，确保冗余。务必通过故障切换测试工具验证这种冗余性。其次，考虑引入合作伙伴多样性，即为不同的连接选择不同的合作伙伴或服务提供商，避免单点故障。第三，在您的本地架构中，确保没有单一故障点，比如冗余的路由器和交换机。对于非关键工作负载，或者带宽需求低于1Gbps的场景，可以考虑将VPN作为备份方案。

### 虚拟接口与链路聚合

Direct Connect的核心在于虚拟接口。这就像在您的本地路由器和AWS之间建立了一个逻辑通道。主要有两种类型：私有虚拟接口和公共虚拟接口。

- 私有VIF用于访问VPC内的资源，使用私有IP地址；
- 公共VIF则用于访问所有AWS公共服务，如S3、Lambda等，使用公共IP地址。

创建VIF时，你需要指定名称、关联的物理连接、VLAN ID、BGP ASN等参数。

![image-20251202210323528](https://s2.loli.net/2025/12/02/wtIHbA7C85QmdFs.png)

为了进一步提升带宽和可靠性，Direct Connect支持链路聚合组LAG。你可以将多个物理Direct Connect连接捆绑成一个逻辑接口。这样做的好处是，不仅增加了总带宽，还提供了内置的冗余。如果其中一个物理连接失效，流量会自动切换到其他连接上，无需手动干预。这对于需要持续高吞吐量的应用场景非常有价值。

![image-20251202210840864](https://s2.loli.net/2025/12/02/KNehxbsvVWZPgqC.png)

高可用性是Direct Connect设计的关键考量。除了前面提到的部署多条专用连接和使用LAG外，还需要关注AWS端的高可用性。例如，一个Direct Connect位置通常会有多个入口点，以提高接入的可靠性。同时，确保您的本地网络设备也具备高可用性，比如使用冗余的路由器和交换机，并正确配置路由协议如BGP，以便在发生故障时能够快速收敛。

![image-20251202210850252](https://s2.loli.net/2025/12/02/kxhnv2I6Rd9blV4.png)

### 安全性和核心组件

安全性方面，对于需要最高级别保护的10G和100Gbps专用Direct Connect连接，AWS提供了MACsec支持。MACsec是在第2层链路层进行加密的标准，符合IEEE 802.1ae。这意味着从您的边缘设备到AWS边缘设备之间的以太网连接本身就被加密了。这种加密是在接近线路速率的速度下进行的，对性能影响很小。而且，这是一个开放标准，得到了包括思科、Juniper、Arista等多家厂商的支持，确保了互操作性。

![image-20251202210934635](https://s2.loli.net/2025/12/02/6F2DEamxh9uPCWw.png)

来梳理一下构建复杂网络连接的核心组件

![image-20251202211003106](https://s2.loli.net/2025/12/02/43Ctn6hUTV9SXFk.png)

- AWS Transit Gateway，它是一个网络中枢，可以轻松地将数千个VPC、AWS账户以及本地网络连接起来
- AWS Direct Connect，它负责在AWS和本地环境之间建立那个私有的、专用的连接。
- AWS Direct Connect Gateway，它的作用是让您通过单个BGP会话就能连接到多个VPC，甚至跨越多个AWS区域。
- Amazon VPC，这是您在AWS云中部署资源的逻辑隔离网络。
- On Premises，也就是您的本地数据中心、办公室或共址设施。

## 实战场景

### Direct Connect Gateway跨区域连接

如何使用AWS Direct Connect Gateway来实现跨区域连接。想象一下，你有一个总部位于美国东部，另一个分支机构在欧洲，都需要访问部署在各自区域的VPC资源。如何高效、经济地实现这种跨洋连接？

<img src="https://s2.loli.net/2025/12/02/XQDOy7SKiPJTnCb.png" alt="image-20251202211132208" style="zoom:67%;" />

核心在于利用Direct Connect Gateway的特性。如图所示，您可以在一个区域比如美国东部建立一个Direct Connect连接，并创建一个Direct Connect Gateway。这个Gateway可以关联到该区域的VPC，也可以关联到其他区域的VPC。您的本地网络通过这个单一的Direct Connect连接，就可以访问所有关联到该Gateway的VPC，无论它们位于哪个区域。这大大简化了跨区域连接的配置。

<img src="https://s2.loli.net/2025/12/02/lYNAKqcTJ1EgR4G.png" alt="image-20251202211629294" style="zoom:67%;" />

上图更清晰地展示了Direct Connect Gateway的作用。它就像一个中央枢纽，可以连接来自不同AWS区域的多个VPC，同时也可以连接到本地网络。**通过一个BGP会话，您的本地路由器就可以学习到所有关联VPC的路由信息，从而实现跨区域的访问。**

![image-20251202211852677](https://s2.loli.net/2025/12/02/CFvzBdpaSDf2AQ5.png)

在这个场景中，我们通常会使用私有虚拟接口。如上图所示，您在本地创建一个私有VIF，将其关联到之前创建的Direct Connect Gateway。**这个VIF需要指定一个VLAN ID和您的本地BGP ASN**。这个私有VIF就像是本地网络通往Direct Connect Gateway的一个入口。

![image-20251202212055123](https://s2.loli.net/2025/12/02/NgKj7Bxlr6Ovq8U.png)

数据是如何流动的呢？当本地发起对某个关联VPC的请求时，数据首先通过本地路由器，经过私有VIF，进入Direct Connect线路，到达AWS侧的Direct Connect Gateway。Gateway根据BGP路由信息，将数据转发到目标VPC。返回的数据流则反向进行。整个过程都是通过私有IP地址完成的，绕过了公共互联网。

![image-20251202212112298](https://s2.loli.net/2025/12/02/XiMwRP7fWETx1pH.png)

关于账户所有权，需要特别注意。Direct Connect Gateway本身是由一个AWS账户比如账户C创建和拥有的。**而连接到这个Gateway的私有VIF，可以由同一个账户创建，也可以由另一个不同的账户比如账户D创建**。这种跨账户的灵活性在大型组织中非常有用，可以实现责任分离和成本归集。

我们来看一下具体的步骤。

![image-20251202212359338](https://s2.loli.net/2025/12/02/rwIotKWTLvkxP2V.png)

第一步，创建DX Gateway。这需要在拥有该Gateway的账户，我们称之为账户C中完成。你需要为这个Gateway指定一个BGP ASN，这个ASN是AWS分配给你的，用于标识这个Gateway。还需要设置一些基本属性，比如名称。

第二步，**创建私有VIF**。这通常在拥有物理Direct Connect连接的账户，我们称之为账户D中完成。在创建VIF时，你需要**指定它要关联到哪个DX Gateway**，也就是账户C中创建的那个。同时，要指定VLAN ID和本地的BGP ASN。这里的关键是，虽然VIF是在账户D创建的，但它的所有权可以指向账户C，或者保持在账户D，具体取决于你的组织结构和策略。

<img src="https://s2.loli.net/2025/12/02/xM4W1dCnjGFmgck.png" alt="image-20251202212417753" style="zoom:67%;" />

继续第二步，配置路由器。由于物理连接通常由账户D拥有，所以需要在账户D的路由器上下载并应用相应的配置，确保它能正确地将流量导向刚刚创建的私有VIF。最后的关键一步接受私有VIF。这需要回到账户C，也就是DX Gateway的所有者。在账户C中，你需要找到刚才创建的私有VIF，然后点击接受。在这个过程中，你需要明确指定这个VIF要连接到哪个DX Gateway，也就是账户C中创建的那个。这样，VIF和Gateway之间的连接就正式建立了。

![image-20251202212844282](https://s2.loli.net/2025/12/02/hFMVRwvkzQ5E6rx.png)

第三步，创建虚拟网关VGW。这需要在拥有目标VPC的账户，我们称之为账户A中完成。VGW是VPC连接到Direct Connect Gateway的桥梁。你需要将VGW附加到具体的VPC上。在这个例子中，我们假设是默认VPC。VGW的BGP ASN在这里其实不那么重要，因为它主要用于内部路由通告。

<img src="https://s2.loli.net/2025/12/02/qzrXcvU1osT9gHl.png" alt="image-20251202212915632" style="zoom: 67%;" />

第四步，将VGW与DX Gateway关联。这同样需要在账户A中进行。你需要指定DX Gateway的ID，也就是账户C中创建的那个。这里还可以选择性地配置一个前缀过滤器，决定哪些VPC CIDR块需要通过DX Gateway通告给本地网络。配置完成后，需要在账户C中再次确认并接受这个关联请求。账户C可以审查并修改前缀过滤器。

![image-20251202213024031](https://s2.loli.net/2025/12/02/UpyGMxsPh9n3SuJ.png)

第五步，也是最后一步，验证BGP路由。你需要检查两个地方的路由表：一个是VPC的路由表，确保它包含了指向本地网络的路由；另一个是本地路由器的路由表，确保它已经学习到了VPC的路由信息。如果两边的路由都正确，那么跨区域的连接就成功了。你可以用ping或者traceroute等工具来实际测试连通性。

### Transit Gateway混合云连接

![image-20251202213122216](https://s2.loli.net/2025/12/02/ByS8zgVjMc2UdXR.png)

如何使用AWS Transit Gateway来实现混合云连接。这个场景更侧重于在本地网络和多个AWS VPC之间建立一个统一的、可扩展的连接架构。混合云连接的**目标是将本地网络与AWS上的多个VPC无缝集成**。Transit Gateway在这里扮演着核心角色，它就像一个中央路由器，可以将各种附件——包括本地的Direct Connect连接、VPC、甚至VPN连接在一起。

![image-20251202213224189](https://s2.loli.net/2025/12/02/Awd1DZeOCV3uEzB.png)

在这个场景中，我们通常会将Transit Gateway与Direct Connect Gateway结合起来使用。如图所示，本地网络通过Direct Connect连接到Transit Gateway。**Transit Gateway内部可以连接多个VPC，无论是同一个账户下的还是不同账户下的。同时，Transit Gateway也可以通过Direct Connect Gateway连接到其他区域的VPC，实现广域的网络互联。**

在这个场景下，我们的**目标是让本地网络能够访问同一个区域内以及跨区域的多个VPC**。这里的关键是**使用Transit VIF，它是专为Transit Gateway设计的虚拟接口类型**。关于账户所有权，虽然可以跨账户部署，但一个常见的模式是：在一个主账户中创建DX、DX Gateway和Transit GW，然后在其他子账户中创建Leaf VPCs，并将它们连接到Transit GW。

在开始配置之前，我们需要做一些准备工作。这包括预先准备好Transit VIF，配置好物理路由器，以及确保已经创建了Direct Connect Gateway，并且Transit VIF已经关联到了这个Gateway。这些准备工作与之前的场景类似，但需要针对Transit Gateway的特点进行调整。

<img src="https://s2.loli.net/2025/12/02/x8kGQ72YdtanZ9V.png" alt="image-20251202213539962" style="zoom:67%;" />

第一步，创建Transit Gateway。你需要为这个Transit GW指定一个BGP ASN。非常重要的一点是，这个ASN不能与之前创建的DX Gateway的ASN重叠。建议在每个AWS区域都使用唯一的ASN。此外，还需要配置Transit GW的默认路由表行为，这会影响后续的路由传播。

<img src="https://s2.loli.net/2025/12/02/19o5l37VNP2HgKr.png" alt="image-20251202213633877" style="zoom:67%;" />

第二步，将Transit GW与DX Gateway关联。这一步是强制性的，不是可选的。**你需要明确指定从Transit GW出发，要通告哪些BGP前缀到DX Gateway，也就是本地网络。**这与之前DX Gateway加VGW场景中的前缀过滤器有所不同。你需要列出你想让本地网络知道的CIDR块，比如172.16.0.0/12。然后，选择你要关联的DX Gateway。

<img src="https://s2.loli.net/2025/12/02/Ey3YOTtFQI1pveP.png" alt="image-20251202213650107" style="zoom:67%;" />

第三步，将VPC连接到Transit GW。你需要选择VPC中合适的子网。Transit GW会在这些子网中创建弹性网络接口ENI，用于与VPC通信。这一步决定了VPC内部的路由走向。你需要指定Transit GW的ID、连接类型为VPC、VPC ID以及要使用的子网ID。

第三步的后续操作是更新VPC的路由表。你需要确保VPC内的路由表指向Transit GW附件，这样才能将发往本地网络或其他VPC的流量导向Transit GW。这通常涉及到添加指向Transit GW附件的路由规则。

<img src="https://s2.loli.net/2025/12/02/RK5BQaP1VonCgHi.png" alt="image-20251202213743619" style="zoom:67%;" />

第四步，验证BGP路由。同样需要检查两个地方的路由表：Transit GW的路由表，确保它包含了正确的路由信息；以及本地路由器的路由表，确保它已经学习到了VPC和Transit GW的路由。如果一切正常，本地网络应该能够通过Transit GW访问到所有已连接的VPC。

## Transit Gateway流量隔离

![image-20251202213821577](https://s2.loli.net/2025/12/02/ze5p36VdKJtlBLD.png)

如何利用Transit Gateway实现流量隔离。这在多租户环境或需要精细化安全策略的场景下尤为重要。Transit Gateway的流量隔离能力主要体现在其路由表功能上。你**可以为Transit Gateway创建多个路由表，并将不同的附件比如VPC、DX连接、VPN连接分配到不同的路由表。**这样，发往某个附件的流量就会被引导到对应的路由表中进行处理。

举个例子，假设你有多个VPC，每个VPC有不同的安全需求。你可以利用Transit Gateway的路由表来实现精细化的流量控制。比如，你可以设置一个专门的路由表，用于流量检测或深度包检测DPI，所有经过这个路由表的流量都会被发送到一个集中的安全设备进行检查。或者，你可以将所有出站到互联网的流量都导向一个中央的NAT网关或防火墙。

大家可以思考一下，如何利用Transit Gateway的路由表来实现类似的安全控制？

![image-20251202213943316](https://s2.loli.net/2025/12/02/KsNTAfZWQvpSdk2.png)

提示一下：关键在于理解和使用Transit GW的路由表功能。Transit GW可以拥有多个路由表。每个附件可以被指定只将流量丢入一个特定的路由表。路由表可以使用任何其他附件作为目标，比如另一个VPC、DX连接或VPN。VPC和DX的CIDR块可以自动传播到路由表中。在这个例子中，你可能不希望默认的路由表关联和传播同时发生，需要手动精细配置。可以尝试在自己的环境中搭建一个类似的实验，用VPN代替DX，用EC2 Linux实例模拟安全设备。