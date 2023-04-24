# BART base with linear attention in encoder VS BART base with vanila attention

All experiments runned on nvidia A100, so if you have less gpu memory i recommend to decrease batch size in 2-3 times and increase grad_acc_steps in the same amount of times. Experiments runned on qnli and mnli datasets, with BART model as base. For running notebooks is enough to clone repository and install requirements in new environment and create kernel for this environment.
Also in extra folder you can find tensorboard logs and submission files for QNLI and MNLI datasets(Unfortunately I was not able to find where I can submit my test of QNLI)



1) The idea for this approach came from a paper titled "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention". The essence of the idea is to    replace the softmax operation, which is typically used to calculate attention weights in transformers, with a simple feature-map-based product. This allows for     linear-time computation of attention weights regardless of sequence length.
However, the problem with this approach is that the vectors for softmax function should ideally be mapped in an infinite space, and using simple mapping functions   like $elu(x)+1$ only provides a rough approximation of softmax attention. As a result, the performance of this approach is slightly worse compared to using softmax.

2) Assuming that $K$, $Q$, and $V$ have the same dimensions of $[S \times h]$, where $S$ is the sequence length and $h$ is the head dimension, we can ignore the $B$ and $H$ dimensions since the calculations are performed independently for each batch and head. Therefore, in both cases, we are dealing with matrices of dimensions $[S \times h]$ (assuming that the mapping is performed in a space with the same dimensions). In softmax attention, first of all, we need to calculate $QK^\intercal$. This matrix multiplication gives us $S \times S \times h$ multiplications and $S \times S \times (h-1)$ additions, so in result, we get $O(S^2 h)$ complexity. On the next step, we should take the softmax of this matrix multiplication and multiply it by $V$, which gives us additional $S \times h \times S$ multiplications and $S \times h \times (S-1)$ additions. However, these additional operations do not change the overall complexity of $O(S^2 h)$. Therefore, the overall complexity for softmax attention is $O(S^2 h)$. For linear attentions hardest operation is sum by j of $K_j V_j^\intercal$ from $j = 1$ to $S$ this operation consists from $h \times h$ multiplications and $S \times h \times h$ additions, in result we get  $O(Sh^2)$ complexity. As result we will get matrix with dimensions $h \times h$ which we need to multiply on Q from left side, this operation also gives us $O(Sh^2)$ complexity, so total calculation complexity is $O(Sh^2)$.
As for the memory complexity for vanila attention we need to store $BH$ attention coeffient matrices with dimensions $S \times S$ and for linear attention we need to store the same amount of matrices but with dimensions $h \times h$ plut S-1 cammulative terms so in resut  $S \times h ^2$ (it is my naive realisation, authors of the article made their own realisation of backprop with cuda drivers and C code and reached $S \times h$ memory consumption.)

3) Seems that i already explained

4) I changed attentions mechanism only in encoder because i did not make any masking mechanism in my code, so it will break decoder because tokens will attend to the future which is not good(not a problem for QNLI and MNLI tasks, just extra caucious). For decoders model without masking is also not good, but not that critical, in that case we will attend to masking tokens MLM or some noising tokens in case of bart(denoising models) but in case of pretrained model it is not a big problem.


5) If h < S we will get benefits in speed, with good implementations(like on the authors github) we also will get benefits in memory. The problem is with masking(as I understand with linear implementations we can make only triangular masking) and another big problem is that this solution is just rough aproximation of softmax, so end results in terms of quality will be worse. Also it is impossible to use attention coefficients for better understanding of model behaviour.

### Big sequence length benchmark, Batch size = 32, emd_dim = 768, heads_qty = 12 (same as in BART)

![Comparison on the big sequence length](https://github.com/DaniilParinov/linear_attn/blob/main/extra/benchmark.png)

### Small sequence length benchmark, Batch size = 32, emd_dim = 768, heads_qty = 12 (same as in BART)

![Comparison on the small sequence length](https://github.com/DaniilParinov/linear_attn/blob/main/extra/benchmark_small_seq_len.png)

On sequence length of size 1000 linear attentions works in approximately 2 times faster then vanila one. To reproduce this graphs run benchmarh.ipynb

## Hit the glue

I decided to choose mnli and qnli datasets. 
Brief descriptions of QNLI:
The QNLI (Question-answering NLI) dataset is a Natural Language Inference dataset automatically derived from the Stanford Question Answering Dataset v1.1 (SQuAD). SQuAD v1.1 consists of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). The dataset was converted into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue. The QNLI dataset is part of GLEU benchmark.
Brief description of MNLI:
The Multi-Genre Natural Language Inference (MultiNLI) corpus is a crowd-sourced collection of 433k sentence pairs annotated with textual entailment information. The corpus is modeled on the SNLI corpus, but differs in that covers a range of genres of spoken and written text, and supports a distinctive cross-genre generalization evaluation. The corpus served as the basis for the shared task of the RepEval 2017 Workshop at EMNLP in Copenhagen.

So by simple words is classification problem, in case of QNLI with 2 labels {0: not_entailment, 1: entailment} and in case of MNLI with 3 labels {0: entailment, 1: neutral, 2:contradiction}, in both cases as metrics used accuracy, MNLI test splited on 2 parts, matched and mistached.

|   | QNLI | Matched MNLI | Mismatched MNLI |
| - | ---- | ------------ | --------------- |
| SOTA |  99.2 |  92 | 91.7  |
| Vanilla | ~91.5(validation)  | 84.25  | 83.38  |
| Linear | ~87.5(validation)  | 80.2  | 80.35  |

|          | Vanilla MNLI | Linear MNLI | Vanilla QNLI | Linear QNLI |
| -------- | ------------ | ----------- | ----------- | ---------- |
| Inference(SPS) |      450        |      450       |          457.5   |        469.7    |
| Train(SPS)    |        534      |      538       |        420.8     |        300    |

As we can see results in terms of speed almost the same, the only outlier is Linear QNLI but i think it was caused by some problems with CPU during running, unfortunately i did not have enough time to run proper profilling
Also it is worth of notice that maximum seq len in bart is limited by 1024 tokes, which is not long enough to see the real difference between linear and vanila attentions

Results of accuracy as expected slightly lower for model with linear attentions, also i could add that my results slihtly undertrained because for bart base was reached performarnce on 2-2.5 points better than in my results.


