import tensorflow as tf
from model.layers import InputBlock, DownsampleBlock, BottleneckBlock, UpsampleBlock, OutputBlock

class Discriminator(tf.keras.Model):
    def __init__(self):
        super.__init__(self)

        self.down_blocks = [DownsampleBlock(filters, idx)
                            for idx, filters in enumerate([16, 32, 64, 128, 128, 128])]
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        out = inputs
        for down_block in self.down_blocks:
            out = down_block(out)

        out = self.dense(out)
        return out