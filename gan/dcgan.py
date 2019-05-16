import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk
from sklearn.utils import shuffle

class DCGAN():
  def __init__(self, img_rows, img_cols, img_channels, dim=100):
    self.dim = dim
    self.img_rows = img_rows
    self.img_cols = img_cols
    self.img_channels = img_channels
    self.img_shape = (img_rows, img_cols, img_channels)
    
    self.generator = self.build_generator()
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss='binary_crossentropy', optimizer=tfk.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    self.combined = self.build_combined()
    self.combined.compile(loss='binary_crossentropy', optimizer=tfk.optimizers.Adam(0.0002, 0.5))
  
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
    valid = tfk.layers.Dense(1, activation='sigmoid')(tensor)
    
    model = tfk.models.Model(inputs=img, outputs=valid)
    #model.summary()
    
    return model
  
  def build_combined(self):
    noise = tfk.Input(shape=(self.dim,), dtype='float32')
    
    img = self.generator(noise)
    self.discriminator.trainable = False
    valid = self.discriminator(img)
    
    model = tfk.models.Model(inputs=noise, outputs=valid)
    
    return model
  
  def train(self, x, epochs, batch_size=128):
    half_batch = int(batch_size / 2)
    steps = -(-x.shape[0] // half_batch)
    
    for epoch in range(epochs):
      x_ = shuffle(x)
      
      for step in range(steps):
        start = step * half_batch
        end = min(start + half_batch, x_.shape[0])
        batch_ = end - start
        
        #train discriminator
        noise = np.random.normal(0, 1, (batch_, self.dim))
        g_images = self.generator.predict(noise)
        r_images = x_[start:end]
        
        d_loss_real = self.discriminator.train_on_batch(r_images, np.ones((batch_, 1)))
        d_loss_fake = self.discriminator.train_on_batch(g_images, np.zeros((batch_, 1)))
        d_loss = np.add(d_loss_real, d_loss_fake) / 2
        
        #train generator
        noise = np.random.normal(0, 1, (batch_size, self.dim))
        g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
        
      print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
      
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
    