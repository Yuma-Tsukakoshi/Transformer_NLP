import tensorflow as tf

class FeedForwardNetwork(tf.keras.models.Model):
  def __init__(self,hidden_dim: int , dropout_rate:float, *args, **kwargs)->None:
    super().__init__(*args, **kwargs)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate
    
    self.filter_dense_layer = tf.keras.layers.Dense(hidden_dim * 4, use_bias=True,activation=tf.nn.relu, name='filter_layer')
    
    self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=True, name='output_layer')
    self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, input: tf.Tensor, training: bool) -> tf.Tensor:
      # →は返値の型を指定してる
        '''
        FeedForwardNetwork を適用します。
        :param input: shape = [batch_size, length, hidden_dim]
        :return: shape = [batch_size, length, hidden_dim]
        '''
        tensor = self.filter_dense_layer(input)
        tensor = self.dropout_layer(tensor, training=training)
        return self.output_dense_layer(tensor)
      
      

class LayerNormalization(tf.keras.layers.Layer):
  '''
  レイヤー正規化という手法を実装したクラスです。レイヤー正規化とは、ニューラルネットワークの各層で、入力データの分布を平均0、分散1になるように正規化することです12。これにより、学習が安定したり、高速化したりする効果が期待できます12。
  '''

  def build(self, input_shape: tf.TensorShape) -> None:
        hidden_dim = input_shape[-1]
        self.scale = self.add_weight('layer_norm_scale', shape=[hidden_dim],initializer=tf.ones_initializer())
        self.bias = self.add_weight('layer_norm_bias', [hidden_dim],initializer=tf.zeros_initializer())
        super().build(input_shape)

  def call(self, x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
      mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
      norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
      
      return norm_x * self.scale + self.bias

class ResidualNormalizationWrapper(tf.keras.models.Model):
  def __init__(self, layer: tf.keras.layer.Layer, dropout_rate: float, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.layer = layer
    self.layer_normalization = LayerNormalization()
    self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    
  def call(self, input: tf.Tensor, training: bool, *args, **kwargs) -> tf.Tensor:
    tensor = self.layer_normalization(input)
    tensor = self.layer(tensor, training=training, *args, **kwargs)
    tensor = self.dropout_layer(tensor, training=training)
    return input + tensor

