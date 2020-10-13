import tensorflow as tf 

class CycleGan(keras.Model):
    def __init__(
        self,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
    ):
        super(CycleGan, self).__init__()
        self.generator_g = generator_g
        self.generator_f = generator_f
        self.discriminator_x = discriminator_x
        self.discriminator_y = discriminator_y
        
    def compile(
        self,
        generator_g_optimizer,
        generator_f_optimizer,
        discriminator_x_optimizer,
        discriminator_y_optimizer,
        generator_loss,
        discriminator_loss,
        calc_cycle_loss,
        identity_loss
    ):
        super(CycleGan, self).compile()
        self.generator_g_optimizer = generator_g_optimizer
        self.generator_f_optimizer = generator_f_optimizer
        self.discriminator_x_optimizer = discriminator_x_optimizer
        self.discriminator_y_optimizer = discriminator_y_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.calc_cycle_loss = calc_cycle_loss
        self.identity_loss = identity_loss

    def call(self, inputs):
        photo_inputs, monet_inputs = inputs
        return self.generator_g(photo_inputs), self.generator_f(monet_inputs)

    @tf.function
    def train_step(self, batch_data):
        real_x, real_y = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
          # Generator G translates X -> Y
          # Generator F translates Y -> X.
          
          fake_y = self.generator_g(real_x, training=True)
          cycled_x = self.generator_f(fake_y, training=True)

          fake_x = self.generator_f(real_y, training=True)
          cycled_y = self.generator_g(fake_x, training=True)

          # same_x and same_y are used for identity loss.
          same_x = self.generator_f(real_x, training=True)
          same_y = self.generator_g(real_y, training=True)

          disc_real_x = self.discriminator_x(real_x, training=True)
          disc_real_y = self.discriminator_y(real_y, training=True)

          disc_fake_x = self.discriminator_x(fake_x, training=True)
          disc_fake_y = self.discriminator_y(fake_y, training=True)

          # calculate the loss
          gen_g_loss = self.generator_loss(disc_fake_y)
          gen_f_loss = self.generator_loss(disc_fake_x)
          
          total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
          
          # Total generator loss = adversarial loss + cycle loss
          total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
          total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

          disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
          disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                              self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                              self.generator_f.trainable_variables)
        
        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                  self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                  self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                  self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                  self.generator_f.trainable_variables))
        
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                      self.discriminator_x.trainable_variables))
        
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                      self.discriminator_y.trainable_variables))

        return {
            "gen_g_loss": total_gen_g_loss,
            "gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss
        }