import tensorflow as tf

class SimpleAttention(tf.keras.models.Model):
  
  def __init__(self, depth: int , *args, **kwargs):
    # depth : 隠れ層及び出力層の次元数
    super().__init__(*args, **kwargs)
    self.depth = depth
    
    # 重みの初期化
    self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="q_dense_layer")
    self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="k_dense_layer")
    self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name="v_dense_layer")
    self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='output_dense_layer')
    
  def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
    q = self.q_dense_layer(input)
    k = self.k_dense_layer(memory)
    v = self.v_dense_layer(memory)
    
    # scaled dot-product attention コンパイラで定義されている変数を利用
    q *= depth ** -0.5
    
    # matmulは行列の内積を計算する関数
    logit = tf.matmul(q, k, transpose_b=True)
    
    # softmax関数を適用することで正規化を行う
    attention_weight = tf.nn.softmax(logit, name="attention_weight")
    
    # 重みをvalueに掛けることで、valueの重み付き平均を計算する
    attention_output = tf.matmul(attention_weight, v)
    return self.output_dense_layer(attention_output)
