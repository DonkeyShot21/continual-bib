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

In fact, DL models are usually haunted by a well studied problem called catastrophic forgetting. Catastrophic forgetting is the tendency of neural models to severely degrade performance on previous tasks when training on new tasks. This is particularly difficult to solve because it is intrinsic to the optmization methods (e.g. gradient descent) used to learn the weights of the networks.

From a Bayesian perspective, avoiding catastrophic forgetting seems straightforward. It is sufficient to retain a distribution over model parameters that indicates the plausibility of any setting given the observed data and then, when new data arrive, combine it with the new information, preventing parameters that strongly influence prediction from changing drastically [[2](#ref2),[3](#ref3)]. The problem with this approach is that Bayesian inference is usually intractable.

Following the idea of restraining drastic changes of the model, some studies introduced regularization terms depending on some Bayesian inference approximations, for instance:  Fisher information [[4](#ewc)], path integral [[5](#ref5)], variational approximation [[3](#ref3)]. To push this concept even further, a few Meta Continual Learning approaches propose to train another neural network to predict parameter update steps instead of trying to formulate a hand-crafted constraint function [[6](#ref6)].

\noindent Other works, instead, make use of extra-memory or generative models that provide a replay of data for past tasks. The result is a cooperative dual model architecture consisting of a \textit{generator} and a \textit{solver} \cite{shin2017continual}. Since the \textit{generator} maximizes the likelihood of generated samples being in the real distribution of data for the previous task, it can be used to feed new data to the \textit{solver}.

\noindent In the next section a novel solution to catastrophic forgetting will be presented, together with a detailed explanation of the architecture of the system, and some insights on how to evaluate its performance.

<a name="ref1"></a>[1] [Investigating Human Priors for Playing Video Games](https://arxiv.org/pdf/1802.10217.pdf), ICML 2018, [website](https://rach0012.github.io/humanRL_website/)<br/>
<a name="ref2"></a>[2] [A Unifying Bayesian View of Continual Learning](https://arxiv.org/abs/1902.06494), NIPS 2018<br/>
<a name="ref3"></a>[3] [Variational Continual Learning](https://arxiv.org/abs/1710.10628), ICLR 2018<br/>
<a name="ref4"></a>[4] [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796), PNAS<br/>
<a name="ref5"></a>[5] [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200), ICML 2017<br/>
<a name="ref6"></a>[6] [Meta Continual Learning](https://arxiv.org/abs/1806.06928), Arxiv

## <a name="datasets"></a>Datasets
| Name | Resolution | Classes | Images | Size | Times Used |
|:-:|:-:|:-:|:-:|:-:|:-:|
| <a name="cifar-10"></a>[MNIST][web:mnist] | 28x28 | 10 | 70k | 20 MB | 0 |
| <a name="cifar-10"></a>[CIFAR-10][web:cifar] | 32x32 | 10 | 60k | 160 MB | 0 |
| <a name="cifar-100"></a>[CIFAR-100][web:cifar] | 32x32 | 100 | 60k | 160 MB | 0 |
| <a name="cifar-1000"></a>[ImageNet-1000][web:imagenet1000] | 469x387* | 1000 | 1.2M | 154 GB | 0 |

[web:mnist]:http://yann.lecun.com/exdb/mnist/
[web:cifar]: https://www.cs.toronto.edu/~kriz/cifar.html
[web:imagenet1000]: http://www.image-net.org/download-images

\* on average, though images are usually downscaled to 256x256

## <a name="formatting"></a>Formatting
Papers are organized in chronological order. Each entry should be formatted as below:

---

<a name="paper_id"></a>**[Name of the Paper][paper:paper_id]**
<br/>
<span style="color:grey">Authors</span><br/>
Name of the conference, Year<br/>

| Category | Code | Inspiration Score |
|:-:|:-:|:-:|
| regularization <br/> sample <br/> generative <br/> meta | [:atom:][code:paper_id] nice code <br/> [:toilet:][code:paper_id] bad code<br/> :no_entry_sign:	 no code | :poop: very bad <br/> :face_with_head_bandage: bad <br/> :neutral_face: ok <br/> :star: good <br/> :fire: very good <br/> :thinking: not sure |

**Summary:**<br/>
three to five lines summary goes here.

**Comment:**<br/>
three to five lines comment goes here.

**Datasets:** list of datasets they use with link to dataset table


[paper:paper_id]: https://arxiv.org
[code:paper_id]: https://github.com

---

## <a name="papers"></a>Papers

- [first paper](#paper_id)
- [second paper](#paper_id)
- [third paper](#paper_id)
