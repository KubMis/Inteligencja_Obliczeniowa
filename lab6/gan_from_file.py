from keras import Sequential
from keras.models import load_model

# zmodyfikowana funkcja definiująca GAN, aby używać załadowanego modelu generatora
def define_gan_with_loaded_generator(d_model, generator_model_path):
    # załaduj zapisany generator
    g_model = load_model(generator_model_path)
    # upewnij się, że wagi dyskryminatora nie są trenowalne
    d_model.trainable = False
    # połącz generator z dyskryminatorem
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # skompiluj GAN
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return g_model, model

# ścieżka do zapisanego generatora
generator_model_path = 'generator_model_010.h5'  # zmień nazwę na odpowiedni plik

# stwórz dyskryminator
d_model = define_discriminator()
# załaduj generator i utwórz GAN
g_model, gan_model = define_gan_with_loaded_generator(d_model, generator_model_path)
# załaduj dane obrazów
dataset = load_real_samples()

# liczba epok
n_epochs = 10
# rozmiar batcha
batch_size = 256

# trenuj model
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, batch_size)
