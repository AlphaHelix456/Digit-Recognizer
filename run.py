from data import load_train, separate_train, preprocess_input
from train import train_model
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

SEED = 42

def plot_confusion_matrix(X, Y, figsize=(10, 6), cmap=plt.cm.Greens):
    Y_pred = model.predict(X)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y, axis=1)
    cm = confusion_matrix(Y_true, Y_pred)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(cm, cmap=cmap, annot=True, square=True)
    ax.set_ylabel('Actual', fontsize=30)
    ax.set_xlabel('Predicted', fontsize=30)
    plt.show()
    

if __name__ == '__main__':
    train_data = load_train()
    X_train, Y_train = separate_train(train_data)
    X_train, Y_train = preprocess_input(X_train, Y_train)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=0.1,
                                                      random_state=SEED)
    
    # train_model(X_train, X_val, Y_train, Y_val)
    model = load_model('model.h5')

    # To test new data, load the data, separate the features from the labels
    # and preprocess the data. Then change X_val and Y_val to the desired data
    final_loss, final_accuracy = model.evaluate(X_val, Y_val, verbose=0)
    print('Final Loss: {:.4f}, Final Accuracy: {:.4f}'.format(
        final_loss, final_accuracy))
    
    plot_confusion_matrix(X_val, Y_val)
    
