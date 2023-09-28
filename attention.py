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
    
  
