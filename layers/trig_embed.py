import tensorflow as tf


class TrigEmbed(tf.keras.layers.Layer):
    """Perform the sinusoidal embeddings for position and directions.
    Args:
        embed_dim: number of sine and cosine functions to compute per coordinate.
        name: Layer Name.
    """
    def __init__(self, embed_dim, name):
        super(TrigEmbed, self).__init__(name=name)
        freq_bands = 2. ** tf.linspace(0., embed_dim - 1, embed_dim)
        period_functions = [tf.math.sin, tf.math.cos]
        self.trig_embedders = [lambda x: x]
        for freq in freq_bands:
            for fn in period_functions:
                self.trig_embedders.append(lambda x, fn=fn,
                                                  freq=freq: fn(x * freq))

    def call(self, inputs):
        return tf.concat([fn(inputs) for fn in self.trig_embedders], -1)
