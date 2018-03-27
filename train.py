from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20


def train_model(X_train, Y_train):
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=0.1,
                                                      random_state=SEED)
    callbacks = get_callbacks()
    datagen = get_datagen(X_train)

    model.fit_generator(datagen.flow(X_Train, Y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=(X_val, Y_val),
                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE)
                        
    model.save('model.h5')

    

def get_callbacks():
    annealer = ReduceLROnPlateau(monitor='val_acc', patience=3,
                             factor=0.5, min_lr=1e-5)
    return [annealer]

def get_datagen(X_train):
    datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1)

    datagen.fit(X_train)
    return datagen
