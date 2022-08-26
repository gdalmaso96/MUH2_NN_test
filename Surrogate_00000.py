import optuna
import time
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import load_model
from multiprocessing.pool import ThreadPool
import subprocess

np.random.seed(299792458)
set_random_seed(299792458)

######### Function DBtonumpy: extracts DB data to be saved in npy files
def DBtonumpy(studyname, database, objective, normalization):
    # Load study
    study = optuna.load_study(study_name=studyname, storage=database)
    
    trials_all = study.get_trials()
    
    # Shuffle data
    np.random.shuffle(trials_all)
    
    trials_p = []
    trials_v = []
    
    for trial in trials_all:
        if(trial.state==optuna.trial.TrialState(1) and trial.values[0] > 0 and trial.values[3] < 80):
            params = trial.params
            params.pop("halfgapL")
            params.pop("halfgapR")
            trials_p.append(params)
            trials_v.append(trial.values[int(objective)])
    
    # Convert data to numpy arrays and rescale between 0 and 1
    X = np.zeros((len(trials_p), len(list(trials_p[0]))))
    Y = np.zeros((len(trials_p), 1))
    
    for i in range(len(trials_p)):
        pars = trials_p[i]
        for j in range(len(list(pars))):
            X[i][j] = pars[list(pars)[j]]
        vals = trials_v[i]
        
        Y[i][0] = vals/normalization
        
    np.save("X.npy", X)
    np.save("Y.npy", Y)



######### Function train: as an input it takes the objective index, and the DB name. It saves the resulting model and plots the learning results
def train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs):
    # Load data
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    
    # Save data batch for test
    X_train, X_test = np.split(X, [int(len(X)*0.8)])
    Y_train, Y_test = np.split(Y, [int(len(X)*0.8)])
    
    # Check if model exists
    if not os.path.exists(studyname + '_%03dneurons_%03dlayers_%05depochs_save' %(nnodes, depth, epochs)):
        # Create model
        model = Sequential()
        model.add(Dense(nnodes, input_shape=(len(list(X[0])), ), activation='tanh'))
        for i in range(depth - 1):
            model.add(Dense(nnodes, activation='tanh'))
        
        model.add(Dense(1))
    else:
        model = load_model(studyname + '_%03dneurons_%03dlayers_%05depochs_save' %(nnodes, depth, epochs))
    
    # Train
    optimiser = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimiser, loss='mse')
    
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batchsize, validation_split=0.25, shuffle=True) #add validate
    
    results = model.evaluate(X_test, Y_test)
    

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.yscale('log')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig('fig/' + studyname + '_%03dneurons_%03dlayers_%05depochstrain_loss.png' %(nnodes, depth, epochs))
    plt.clf()
    
    Y_pred = model.predict(X_train)
        
    plt.plot(Y_pred, Y_train, '.')
    plt.plot([0, 1], [0, 1], '--', color = 'r')
    plt.title('Prediction on train')
    plt.ylabel('Train value')
    plt.xlabel('Prediction')
    #plt.show()
    plt.savefig('fig/' + studyname + '_%03dneurons_%03dlayers_%05depochstrain.png' %(nnodes, depth, epochs))
    plt.clf()
    
    
    # Save model
    model.save(studyname + '_%03dneurons_%03dlayers_%05depochs_save' %(nnodes, depth, epochs))
    
    # Create test plot
    Y_pred = model.predict(X_test)
    
    plt.plot(Y_pred, Y_test, '.')
    plt.plot([0, 0.2], [0, 0.2], '--', color = 'r')
    plt.title('Prediction on test')
    plt.ylabel('Test value')
    plt.xlabel('Prediction')
    #plt.show()
    plt.savefig('fig/' + studyname + '_%03dneurons_%03dlayers_%05depochstest.png' %(nnodes, depth, epochs))
    plt.clf()
    
    return results

if __name__ == '__main__':
    studyname = 'MUH2_V5_Peter_TOT_mup_mono'
    database = 'sqlite:///' + studyname + '.db'
    objective = 0
    batchsize = 500
    normalization = (2922072/2e11*2.4e-3/1.6e-19)*0.2 # 20% of total muons
    nnodes = 20
    depth = 1
    epochs = 10000
    start_time = time.time()
    #DBtonumpy(studyname, database, objective, normalization)
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 1 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 2 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 3 layer
    print(time.time() - start_time)
    depth += 2
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 5 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 6 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 7 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 8 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 9 layer
    print(time.time() - start_time)
    depth += 1
    train(studyname, database, objective, batchsize, normalization, nnodes, depth, epochs) # 10 layer
    
