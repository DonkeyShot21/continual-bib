# Inspiring Continual Learning
A curated list of inspiring Continual Learning (aka Incremental Learning) resources.

## <a name="intro"></a>Introduction

It is a well known fact that biological brains considerably outperform current artificial neural networks when it comes to learning adaptively. Humans, as most of life forms, are, in fact, orders of magnitude more efficient than our Deep Learning (DL) models at tackling never seen problems sequentially. There is much evidence that this is due to our ability to exploit previously learned priors [[1](#ref1)] to understand the mechanics behind the new task, in order to quickly come to a solution.

This is possible because our brain is able to learn continually, acquiring knowledge while also retaining useful information about previous experience. In the context of Machine Learning (ML), this capability is generally called Continual learning (CL).

More specifically, an artificial CL agent should also exhibit the following attributes:
- **Online learning**: the model should be able to update itself using a sequential stream of data or tasks;
- **Presence of transfer**: knowledge of old task should help improve performance on new ones and vice versa;
- **Bounded system size**: the capacity of the model in terms of number of neurons and available memory should be fixed;
- **No direct access to previous experience**.

Arguably, CL is one of the ingredients that play a fundamental role in the creation of Artificial General Intelligence (AGI). Unfortunately, it just so happens that the current generation of neural networks performs very poorly in this adaptive scenario.

In fact, DL models are usually haunted by a well studied problem called **catastrophic forgetting**. Catastrophic forgetting is the tendency of neural models to severely degrade performance on previous tasks when training on new tasks. This is particularly difficult to solve because it is intrinsic to the optmization methods (e.g. gradient descent) used to learn the weights of the networks.

From a Bayesian perspective, avoiding catastrophic forgetting seems straightforward. It is sufficient to retain a distribution over model parameters that indicates the plausibility of any setting given the observed data and then, when new data arrive, combine it with the new information, preventing parameters that strongly influence prediction from changing drastically [[2](#ref2),[3](#ref3)]. The problem with this approach is that Bayesian inference is usually intractable.

Following the idea of restraining drastic changes of the model, some studies introduced regularization terms depending on some Bayesian inference approximations, for instance:  Fisher information [[4](#ewc)], path integral [[5](#ref5)], variational approximation [[3](#ref3)]. To push this concept even further, a few Meta Continual Learning approaches propose to train another neural network to predict parameter update steps instead of trying to formulate a hand-crafted constraint function [[6](#ref6)].

Other works, instead, make use of extra-memory or generative models that provide a replay of data for past tasks. The result is a cooperative dual model architecture consisting of a *generator* and a *solver* [[7](#ref7)]. Since the *generator* maximizes the likelihood of generated samples being in the real distribution of data for the previous task, it can be used to feed new data to the *solver*.

<a name="ref1"></a>[1] [Investigating Human Priors for Playing Video Games](https://arxiv.org/pdf/1802.10217.pdf), ICML 2018, [website](https://rach0012.github.io/humanRL_website/)<br/>
<a name="ref2"></a>[2] [A Unifying Bayesian View of Continual Learning](https://arxiv.org/abs/1902.06494), NIPS 2018<br/>
<a name="ref3"></a>[3] [Variational Continual Learning](https://arxiv.org/abs/1710.10628), ICLR 2018<br/>
<a name="ref4"></a>[4] [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796), PNAS<br/>
<a name="ref5"></a>[5] [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200), ICML 2017<br/>
<a name="ref6"></a>[6] [Meta Continual Learning](https://arxiv.org/abs/1806.06928), Arxiv <br/>
<a name="ref7"></a>[7] [Continual learning with deep generative replay](https://arxiv.org/abs/1705.08690), NIPS 2017

## <a name="datasets"></a>Datasets
| Name | Resolution | Classes | Images | Size | Times Used |
|:-:|:-:|:-:|:-:|:-:|:-:|
| <a name="mnist"></a>[MNIST][web:mnist] | 28x28 | 10 (permuted / disjoint) | 70k | 20 MB | 1 |
| <a name="cifar"></a>[CIFAR][web:cifar] | 32x32 | 10 / 100 | 60k | 160 MB | 1 |
| <a name="imagenet-1000"></a>[ImageNet][web:imagenet] | 469x387* | 1000 | 1.2M | 154 GB | 1 |

[web:mnist]: http://yann.lecun.com/exdb/mnist/
[web:cifar]: https://www.cs.toronto.edu/~kriz/cifar.html
[web:imagenet]: http://www.image-net.org/download-images

\* on average, though images are usually downscaled to 256x256

## <a name="template"></a>Template
Each entry should be formatted as below:

---

<a name="paper_id"></a>[Title of the Paper][paper:paper_id], Conference <br/>
*Authors*<br/>

| Category | Datasets | Code | Inspiration Score |
|:-:|:-:|:-:|:-:|:-:|
| regularization <br/> sample <br/> generative <br/> meta | list of datasets | [<img src="icons/pytorch.png" alt="pytorch" height="24"/>][code:paper_id] pytorch <br/> [<img src="icons/tensorflow.png" alt="tensorflow" height="24"/>][code:paper_id] tensorflow <br/> :no_entry_sign: no code | :poop: very bad <br/> :face_with_head_bandage: bad <br/> :neutral_face: ok <br/> :star: good <br/> :fire: very good <br/> :thinking: not sure |

**Summary:**<br/>
three to five lines summary.

**Comment:**<br/>
three to five lines comment.

[paper:paper_id]: https://arxiv.org
[code:paper_id]: https://github.com

---
<!--- an example, copy paste from here

---

<a name="paper_id"></a>[Name of the Paper][paper:paper_id], conference <br/>
*Authors*<br/>

| Category | Datasets | Code | Inspiration Score |
|:-:|:-:|:-:|:-:|
| regularization | [disjoint-MNIST](#disjoint-mnist)  | [<img src="icons/pytorch.png" height="24"/>](github.com) | :star: |


**Summary:**<br/>
three to five lines summary goes here.

**Comment:**<br/>
three to five lines comment goes here.

[paper:paper_id]: https://arxiv.org
[code:paper_id]: https://github.com

---

-->

## <a name="papers"></a>Papers

Papers are organized in chronological order.

#### Index

- [Learning without Forgetting](#lwf), ECCV 2016

#### Details

---

<a name="lwf"></a>[Learning without Forgetting](https://arxiv.org/abs/1606.09282), ECCV 2016<br/>
*Zhizhong Li, Derek Hoiem*

| Category | Datasets | Code | Inspiration Score |
|:-:|:-:|:-:|:-:|
| regularization | MNIST, CIFAR, ImageNet, ... | [<img src="https://upload.wikimedia.org/wikipedia/commons/2/21/Matlab_Logo.png" height="24"/>](https://github.com/lizhitwo/LearningWithoutForgetting#installation) [<img src="icons/pytorch.png" height="24"/>](https://github.com/GMvandeVen/continual-learning) | :star: |


**Summary:**<br/>
Foundational work on CL, LwF proposes to preserve the performance on the old task using [knowledge distillation](https://arxiv.org/abs/1503.02531). They introduce a regularization term (distillation loss) in training that encourages the outputs of the new network to approximate the outputs of the old network. Several regularization losses (L1, L2, cross-entropy) are tested with similar results to distillation loss. Both single and multiple new tasks are explored. The experiments show that LwF moderately outperforms feature extraction, finetuning, [Less-forgetting Learning](#lfl), while compared to the joint training setup, it tends to underperform on the old task (as expected).

**Comment:**<br/>
The main disadvantage of LwF is that it seems to work only in the case that the two tasks (new and old) are very similar. Also, this approach is expensive as it requires computing a forward pass through the old task’s network for every new data point.
