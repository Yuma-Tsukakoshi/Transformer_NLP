# 1.sample Attention の理解

![simple Attentionの概要図](https://camo.qiitausercontent.com/0df89d309e385fb9b47b74b6fd81833f2c9f5b14/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f34393736316432632d376436382d303139392d353863612d3833333430353330383134312e706e67)

![simple Attentionの論文バージョン](https://camo.qiitausercontent.com/9b8af7118dcd4c006bc531f105f969b1da00dbc9/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f38626631643534342d663834322d343336322d303365392d3333343533333338363563362e706e67)

## 2 種類の Attention の使い方がある

- 1.self-attention
  query と memory が一致する x , x  
  Self-Attention は言語の文法構造であったり、照応関係（its が指してるのは Law だよねとか）を獲得するのにも使われている
  Self Attention は簡単に言うと「離れた所も畳み込める CNN」の様なものです。

- 2.SouceTarget-Attention
  query と memory が一致しない x , y  
  SourceTarget-Attention は Transformer では Decoder で使われます。
  Decoder はある時刻 t のトークンを受け取って t+1 の時刻のトークンを予測します。

# 2.Scaled Dot-production の理解

Softmax 関数は、 logit の値が大きいと値が飽和してしまい、 gradient が 0 に近くなってしまう。
Softmax の logit は query と key の行列積です。従って、 query, key の次元（depth）が大きいほど logit は大きくなります。

そこで、 query の大きさを depth に従って小さくしてあげます。

```math
attention\_weight = softmax(\frac{qk^T}{\sqrt{depth}})

```

### MASKを用いる
attention_weight を0にしたい場合は次の２つがある。
⇒softmax関数において,-∞にすることで0にする

- PAD を無視したい ⇒ トークン列の長さを統一させるための処理  
PADによって計算処理の結果が変わってしまう⇒そのため、PADは無視する必要がある。

- Decoder の Self-Attention で未来の情報を参照できないようにしたい  
![maskにおいて未来の情報を参照できないようにする](https://camo.qiitausercontent.com/c1c0194389ab9dbf3831853913dc15b5710255cc/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f38646161313233372d623336392d663233342d376134332d6233356539383634383837392e706e67)

## Multi-head Attention
仕組み：  
query, key, value をそれぞれ head_num 個に split してからそれぞれ attention を計算し、最後に concat するだけです。
この Multi-head Attention が RNN における LSTM, GRU セルのように Attention ベースのモデルの基本単位になってきます。

![Multi-head Attentionの仕組み図](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F61079%2F5a964f15-4997-e9e7-c13e-9aeab8ea1a61.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=e340f30c512c21effe2ea94614058470)

何度も繰り返し適用することでより複雑な学習ができるようになる
⇒Hopping  


