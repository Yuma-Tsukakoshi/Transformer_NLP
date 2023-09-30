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
    # mask は pad 部分などが1, 他は0 0にしたい部分がTrueに変更⇒-∞の値になる
    logit += tf.to_float(attention_mask) * input.dtype.min
    
    # softmax関数を適用することで正規化を行う
    attention_weight = tf.nn.softmax(logit, name="attention_weight")
    
    # 重みをvalueに掛けることで、valueの重み付き平均を計算する
    attention_output = tf.matmul(attention_weight, v)
    return self.output_dense_layer(attention_output)
  
class MultiheadAttention(tf.keras.models.Model):
    '''
    Multi-head Attention のモデルです。
    model = MultiheadAttention(
        hidden_dim=512,
        head_num=8,
        dropout_rate=0.1,
    )
    model(query, memory, mask, training=True)
    '''
    
    def __init__(self,hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
        '''
        コンストラクタです。
        :param hidden_dim: 隠れ層及び出力の次元
            head_num の倍数である必要があります。
        :param head_num: ヘッドの数
        :param dropout_rate: ドロップアウトする確率
        '''
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        
        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='v_dense_layer')
        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self,
            input: tf.Tensor,
            memory: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool,) -> tf.Tensor:
        '''
        モデルの実行を行います。
        :param input: query のテンソル
        :param memory: query に情報を与える memory のテンソル
        :param attention_mask: attention weight に適用される mask
            shape = [batch_size, 1, q_length, k_length] のものです。
            pad 等無視する部分が True となるようなものを指定してください。
        :param training: 学習時か推論時かのフラグ
        '''
        
        q = self.q_dense_layer(input)  # [batch_size, q_length, hidden_dim]
        k = self.k_dense_layer(memory)  # [batch_size, m_length, hidden_dim]
        v = self.v_dense_layer(memory)

        q = self._split_head(q)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        k = self._split_head(k)  # [batch_size, head_num, m_length, hidden_dim/head_num]
        v = self._split_head(v)  # [batch_size, head_num, m_length, hidden_dim/head_num]
        
        # ここでhead_numで割ってることでバッチとして分割してる
        depth = self.hidden_dim // self.head_num
        q *= depth ** -0.5
        
        logit = tf.matmul(q,k,transpose_b=True)
        logit += tf.to_float(attention_mask) * input.dtype.min
        
        attention_weight = tf.nn.softmax(logit,name='attention_weight')
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)
        
        # 重みに従って value から情報を引いてきます
        attention_output = tf.matmul(attention_weight, v)  # [batch_size, head_num, q_length, hidden_dim/head_num]
        # ここでコンバインして結合する
        attention_output = self._combine_head(attention_output)  # [batch_size, q_length, hidden_dim]
        return self.output_dense_layer(attention_output)    
      
    def _split_head(self, x:tf.Tensor) -> tf.Tensor:
        '''
        入力の tensor の hidden_dim の次元をいくつかのヘッドに分割します。
        入力 shape: [batch_size, length, hidden_dim] の時
        出力 shape: [batch_size, head_num, length, hidden_dim//head_num]
        となります。
        '''
        
        # tf.name_scopeという関数は、テンソルや演算に名前を付ける
        with tf.name_scope('split_head'):
          # tf.unstackという関数で、そのテンソルを軸ごとに分解してリストにする
          batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
          x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
          # 元々の形状が[batch_size, length, head_num, hidden_dim//head_num]であるテンソルを、[batch_size, head_num, length, hidden_dim//head_num]という形状にする。
          return tf.transpose(x, [0, 2, 1, 3])
    
    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        '''
        入力の tensor の各ヘッドを結合します。 _split_head の逆変換です。
        入力 shape: [batch_size, head_num, length, hidden_dim//head_num] の時
        出力 shape: [batch_size, length, hidden_dim]
        となります。
        '''
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])


class selfAttention(MultiheadAttention):
  def call(
    self,
    input: tf.Tensor,
    attention_mask: tf.Tensor,
    training: bool,
  ) -> tf.Tensor:
    # 親クラスのcallメソッドを呼び出している
      return super().call(
            input=input,
            memory=input,
            attention_mask=attention_mask,
            training=training,
        )
