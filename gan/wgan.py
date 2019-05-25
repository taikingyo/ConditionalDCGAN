import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk
from sklearn.utils import shuffle
from tensorflow.keras import backend as K

class WGANgp():
  def __init__(self,
               img_rows,
               img_cols,
               img_channels,
               dim=100,
               batch_size=32,
               gradient_penalty_weight=10):
    self.img_rows = img_rows
    self.img_cols = img_cols
    self.img_channels = img_channels
    self.img_shape = (img_rows, img_cols, img_channels)
    self.dim = dim
    self.batch_size = batch_size
    self.gp_weight = gradient_penalty_weight
    
    self.generator = self.build_generator()
    self.discriminator = self.build_discriminator()
    
    #generator学習時、discriminatorの学習をOFF
    for layer in self.discriminator.layers:
      layer.trainable = False
    self.discriminator.trainable = False
    
    self.G_model, self.G_train = self.build_combined()
    
    #discriminator学習時、generatorの学習をOFF
    for layer in self.discriminator.layers:
      layer.trainable = True
    self.discriminator.trainable = True
    
    for layer in self.generator.layers:
      layer.trainable = False
    self.generator.trainable = False
    
    self.D_train = self.discriminator_with_own_loss()
  
  def build_generator(self):
    f1, f2 = self.common_factor(self.img_rows, self.img_cols)
    n_filter = 64 * 2 ** (len(f1) - 2)
    
    noise = tfk.Input(shape=(self.dim,), dtype='float32', name='img_seed')
    
    tensor = tfk.layers.Dense(n_filter * f1[-1] * f2[-1])(noise)
    tensor = tfk.layers.Reshape((f1[-1], f2[-1], n_filter))(tensor)
    tensor = tfk.layers.BatchNormalization()(tensor)
    n_filter = n_filter // 2
    f1 = f1[:-1]
    f2 = f2[:-1]
    
    for _ in range(len(f1) - 1):
      tensor = tfk.layers.Conv2DTranspose(n_filter, 5, f1[-1], padding='same', activation='relu', kernel_initializer='glorot_normal')(tensor)
      tensor = tfk.layers.BatchNormalization()(tensor)
      n_filter = n_filter // 2
      f1 = f1[:-1]
      f2 = f2[:-1]
    
    images = tfk.layers.Conv2DTranspose(self.img_channels, 5, strides=(f1[0], f2[0]), padding='same', activation='tanh', kernel_initializer='glorot_normal')(tensor)
    
    model = tfk.models.Model(inputs=noise, outputs=images)
    #model.summary()
    
    return model
  
  def build_discriminator(self):
    f1, f2 = self.common_factor(self.img_rows, self.img_cols)
    h = self.img_rows
    w = self.img_cols
    n_filter = 64
    
    img = tfk.Input(shape=self.img_shape, dtype='float32', name='image')
    
    tensor = img
    
    for _ in range(len(f1) - 1):
      tensor = tfk.layers.Convolution2D(n_filter, kernel_size=5, strides=(f1[0], f2[0]), padding='same', kernel_initializer='glorot_normal')(tensor)
      tensor = tfk.layers.LeakyReLU(0.2)(tensor)
      n_filter *= 2
      f1 = f1[1:]
      f2 = f2[1:]
      
    tensor = tfk.layers.Flatten()(tensor)
    tensor = tfk.layers.Dense(256)(tensor)
    tensor = tfk.layers.LeakyReLU(0.2)(tensor)
    tensor = tfk.layers.Dropout(0.5)(tensor)
    valid = tfk.layers.Dense(1)(tensor)
    
    model = tfk.models.Model(inputs=img, outputs=valid)
    #model.summary()
    
    return model
  
  def discriminator_with_own_loss(self):
    z = tfk.Input(shape=(self.dim,))
    
    e = K.placeholder(shape=(None,1,1,1))
    f_img = self.generator(z)
    r_img = tfk.Input(shape=(self.img_shape))
    a_img = tfk.Input(shape=(self.img_shape), tensor=e * r_img + (1 - e) * f_img)
    
    f_out = self.discriminator(f_img)
    r_out = self.discriminator(r_img)
    a_out = self.discriminator(a_img)
    
    loss_real = K.mean(r_out) / self.batch_size
    loss_fake = K.mean(f_out) / self.batch_size
    
    grad_mixed = K.gradients(a_out, [a_img])[0]
    norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
    grad_penalty = K.mean(K.square(norm_grad_mixed - 1))
    
    loss = loss_fake - loss_real + self.gp_weight * grad_penalty
    
    training_updates = tfk.optimizers.Adam(lr=1e-4,
                                           beta_1=0.5,
                                           beta_2=0.9).get_updates(loss, self.discriminator.trainable_weights)
    
    d_train = K.function([r_img, z, e], [loss_real, loss_fake], training_updates)
    return d_train
  
  def build_combined(self):
    z = tfk.Input(shape=(self.dim,))
    img = self.generator(z)
    valid = self.discriminator(img)
    model = tfk.models.Model(z, valid)
    #model.summary()
    
    loss = -1. * K.mean(valid)
    training_updates = tfk.optimizers.Adam(lr=1e-4,
                                           beta_1=0.5,
                                           beta_2=0.9).get_updates(loss, self.generator.trainable_weights)
    
    g_train = K.function([z], [loss], training_updates)
    return model, g_train
  
  def train(self,
            x,
            epochs,
            training_ratio=5):
    
    steps = -(-x.shape[0] // self.batch_size)
    
    for epoch in range(epochs):
      x_ = shuffle(x)
      
      for step in range(steps):
        start = step * self.batch_size
        end = min(start + self.batch_size, x_.shape[0])
        batch_ = end - start
        
        #train discriminator
        noise = np.random.normal(0, 1, (batch_, self.dim))
        epsilon = np.random.uniform(size=(batch_, 1, 1, 1))
        loss_real, loss_fake = self.D_train([x_[start:end], noise, epsilon])
        d_loss = loss_real - loss_fake
        
        if step % training_ratio == 0:
          #train generator
          noise = np.random.normal(0, 1, (batch_, self.dim))
          g_loss = self.G_train([noise])
        
      if epoch % 10 == 0:
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss[0]))
      
    self.show_imgs(100)
        
  def show_imgs(self, seed=None):
    r, c = 5, 5
    if seed is not None:
      np.random.seed(seed)
    noise = np.random.normal(0, 1, (r * c, self.dim))
    gen_imgs = self.generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    if(self.img_channels == 1):
      gen_imgs = gen_imgs.reshape([r * c, self.img_rows, self.img_cols])
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt])
        axs[i, j].axis('off')
        cnt += 1
        
  def common_factor(self, x1, x2):
    #自動調整の為の公約数の計算
    f1 = []
    f2 = []
    
    b = 2
    while b < x1 and b < x2:
      if x1 % b == 0 and x2 % b == 0:
        f1.append(b)
        f2.append(b)
        x1 = x1 // b
        x2 = x2 // b
      else:
        b += 1
    f1.append(x1)
    f2.append(x2)
    
    return f1, f2
    