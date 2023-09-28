# 1.sample Attentionの理解

![simple Attentionの概要図](https://camo.qiitausercontent.com/0df89d309e385fb9b47b74b6fd81833f2c9f5b14/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f34393736316432632d376436382d303139392d353863612d3833333430353330383134312e706e67)

![simple Attentionの論文バージョン](https://camo.qiitausercontent.com/9b8af7118dcd4c006bc531f105f969b1da00dbc9/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f38626631643534342d663834322d343336322d303365392d3333343533333338363563362e706e67)

## 2種類のAttentionの使い方がある
- 1.self-attention
queryとmemoryが一致する x , x  
Self-Attention は言語の文法構造であったり、照応関係（its が指してるのは Law だよねとか）を獲得するのにも使われている
Self Attentionは簡単に言うと「離れた所も畳み込めるCNN」の様なものです。



- 2.SouceTarget-Attention
queryとmemoryが一致しない  x , y  
SourceTarget-Attention は Transformer では Decoder で使われます。
Decoder はある時刻 t のトークンを受け取って t+1 の時刻のトークンを予測します。


# 2.Scaled Dot-productionの理解
Softmax 関数は、 logit の値が大きいと値が飽和してしまい、 gradient が0に近くなってしまう。
Softmax の logit は query と key の行列積です。従って、 query, key の次元（depth）が大きいほど logit は大きくなります。

そこで、 query の大きさを depth に従って小さくしてあげます。

```math
attention\_weight = softmax(\frac{qk^T}{\sqrt{depth}})

```
