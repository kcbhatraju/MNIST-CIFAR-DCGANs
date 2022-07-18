import tensorflow as tf
import matplotlib.pyplot as plt

def gen_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8*8*256,use_bias=False,input_shape=[100,]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((8,8,256)),
        tf.keras.layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding="same",use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding="same",use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding="same",use_bias=False),
    ])
    
    model.filename = "generator"
    return model

def discrim_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(5,5),strides=(2,2),padding="same",input_shape=[32,32,3]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128,(5,5),strides=(2,2),padding="same",input_shape=[32,32,3]),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    
    model.filename = "discriminator"
    return model

crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def calc_gen_loss(fake_output):
    return crossentropy(tf.ones_like(fake_output), fake_output)

def calc_discrim_loss(real_output, fake_output):
    real_loss = crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = crossentropy(tf.ones_like(fake_output), fake_output)
    return real_loss + fake_loss

gen_optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
discrim_optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
gen, discrim = gen_model(), discrim_model()


class Checkpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save(f"{self.model.filename}")

epochs = 50
noise_dim = 100
example_dim = 16
buffer = 50000
batch_size = 256
seed = tf.random.normal([example_dim, noise_dim])

@tf.function
def train_step(real_imgs):
    noise = tf.random.normal([example_dim, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discrim_tape:
        gen_imgs = gen(noise, training=True)
        real_output = discrim(real_imgs, training=True)
        fake_output = discrim(gen_imgs, training=True)
        
        gen_loss = calc_gen_loss(fake_output)
        discrim_loss = calc_discrim_loss(real_output, fake_output)
        
        gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
        discrim_grad = discrim_tape.gradient(discrim_loss, discrim.trainable_variables)
        
        gen_optim.apply_gradients(zip(gen_grad, gen.trainable_variables))
        discrim_optim.apply_gradients(zip(discrim_grad, discrim.trainable_variables))

def train(epochs):
    cifar = tf.keras.datasets.cifar10
    (train_imgs, train_labs), (_, _) = cifar.load_data()

    # classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    train_imgs = train_imgs / 255
    norm_mean, norm_std = train_imgs.mean(), train_imgs.std()
    train_imgs = (train_imgs - norm_mean) / norm_std
    train_set = tf.data.Dataset.from_tensor_slices(train_imgs).shuffle(buffer).batch(batch_size)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in train_set:
            train_step(batch)
        
        test_imgs = gen(seed, training=False)
        _ = plt.figure(figsize=(4,4))
        
        if (epoch+1) % 10 == 0:
            for i in range(test_imgs.shape[0]):
                plt.subplot(4,4, i+1)
                plt.imshow(test_imgs[i,:,:,:]*norm_std+norm_mean)
                plt.axis("off")
            
            plt.show()

train(epochs)