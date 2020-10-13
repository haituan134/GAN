import tensorflow as tf 
from model.generator import Unet
from model.discriminator import Discriminator

class Gan(tf.keras.Model):
    def __init__(self):
        super().__init__(self)
        self.generator = Unet
        self.discriminator = Discriminator

    def call(self, inputs):
        return self.generator(inputs)

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        return super().compile(self)
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    @tf.function
    def train_step(self, data):
        inputs, targets = data

        with tf.GradientTape(persistent=True) as tape: 
            generated = self.generator(inputs)

            real_output = self.discriminator(targets)
            fake_output = self.discriminator(generated)

            generator_loss = self.generator_loss(fake_output)
            discriminator_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradient(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradient(zip(gradients_of_discriminator, self.discriminator.trainable_variables))