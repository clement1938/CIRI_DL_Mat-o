# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:26:05 2024

@author: 33695
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import scipy.special as sp


#%%


#%%
# from numba import jit, njit
# from multiprocessing import Pool

# def execute_task(args):
#     # Votre code ici
#     return result

# if __name__ == '__main__':
#     with Pool(processes=4) as pool:
#         results = pool.map(execute_task, list_of_args)

#%% Partie réseau neuronaux

def initialisation(dimensions):
    
    parametres = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres


def forward_propagation(X, parametres): # , activation):
  
  activations = {'A0': X}

  C = len(parametres) // 2

  for c in range(1, C + 1):

    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    # if activation == 'sigmoid':
    #     activations['A' + str(c)] = sigmoid(Z)
    # if activation == 'ReLU':
    #     activations['A' + str(c)] = ReLU(Z)

  return activations


def back_propagation(y, parametres, activations):

  m = y.shape[1]
  C = len(parametres) // 2

  dZ = activations['A' + str(C)] - y
  gradients = {}

  for c in reversed(range(1, C + 1)):
    gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
    gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
      dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

  return gradients


def update(gradients, parametres, learning_rate):

    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parametres


def predict(X, parametres):
  activations = forward_propagation(X, parametres)
  C = len(parametres) // 2
  Af = activations['A' + str(C)]
  return Af >= 0.5


def deep_neural_network(X, y, hidden_layers = (128, 64, 32), learning_rate = 0.005, epochs = 2**10, sigma = 0):
    
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(epochs), 2))

    C = len(parametres) // 2

    m,n = X.shape

    # gradient descent
    for i in tqdm(range(epochs)):
        
        # lr = update_learning_rate(learning_rate, i, decay_rate=0.001)
        
        X_bruite = X + np.random.normal(0, sigma, (m,n))             # ajout du bruit Gaussien à chauque itération
        activations = forward_propagation(X_bruite, parametres)
        gradients = back_propagation(y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
        y_pred = predict(X, parametres)
        training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    return training_history, parametres


#%% Partie telecom

# Step 1: Define the necessary parameters for our simulation
def encodage_matrice(bits, G, k):
    bits_reshaped = bits.reshape(len(bits)//k, k)
    bits_encodes = bits_reshaped @ G % 2
    return bits_encodes.flatten()

def mat_combi(G, k):
    n_combinations = 2**k
    matrice_combinaisons = np.zeros((n_combinations, k), dtype=int)
    for i in range(n_combinations): # Rempli la matrice avec des combinaisons de 8 bits
        # Convertir l'entier i en une chaîne binaire de 8 caractères, puis en un tableau de bits
        binary_string = format(i, '08b')  # Format i as a binary string with padding
        bit_array = np.array([int(bit) for bit in binary_string])
        matrice_combinaisons[i, :] = bit_array
    return (matrice_combinaisons@G)%2

def mat():
    n_combinations = 2**8
    matrice_combinaisons = np.zeros((n_combinations, k), dtype=int)
    for i in range(n_combinations): # Rempli la matrice avec des combinaisons de 8 bits
        # Convertir l'entier i en une chaîne binaire de 8 caractères, puis en un tableau de bits
        binary_string = format(i, '08b')  # Format i as a binary string with padding
        bit_array = np.array([int(bit) for bit in binary_string])
        matrice_combinaisons[i, :] = bit_array
    return matrice_combinaisons

# BPSK Modulation Function
def BPSK_modulation(bits):  # A revoir pour envoyer des entiers
    return (2*bits - 1).astype(int)

# Threshold Detection for BPSK
def threshold_detection(received_signals): #################################################
    return received_signals > 0            #.5 pour le réseau de neurones ?

# Counting the number of bit errors
def count_errors(original_bits, detected_bits):
    return np.sum(original_bits != detected_bits)

# Theoretical BER calculation using Q-function
def theoretical_ber(Eb_No):
    return 0.5 * sp.erfc(np.sqrt(Eb_No))

def result_comp(received_signals, G, MAT): ################################################### a voir si on la laisse pour comparer avec les reseaux de neuronnes
    lignes_correspondante = []
    for i in range(len(received_signals)//(2*k)):
        distances = np.sum(np.abs(M - received_signals[16*i:16*(i+1)]), axis=1)
        indice_min = np.argmin(distances) # Trouve l'indice de la ligne avec la distance minimale (max de vraisemblance)
        ligne_correspondante = MAT[indice_min, :] # Récupère la ligne correspondante de la matrice_combinaisons
        lignes_correspondante.append(ligne_correspondante)
    return np.array(lignes_correspondante) # estimé


def add_noise(signals, sigma):
    noise = np.random.normal(0, sigma, signals.shape)
    return signals + noise

# def train_neural_network(X_train, Y_train, X_test, Y_test, noise_level, epochs=2**10, learning_rate=0.5):  # training

#     # Initialize the model
#     model = Sequential([
#         Dense(16, input_dim=X_train.shape[1], activation='ReLU'),
#         Dense(128, activation='ReLU'),
#         Dense(64, activation='ReLU'),
#         Dense(32, activation='ReLU'),
#         Dense(Y_train.shape[1], activation='sigmoid')
#     ])

#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

#     # Add noise to training data
#     X_train_noisy = add_noise(X_train, noise_level)

#     # Train the model
#     model.fit(X_train_noisy, Y_train, epochs=epochs, verbose=0)

#     # Evaluate the model
#     _, accuracy = model.evaluate(X_test, Y_test)
#     print(f"Accuracy: {accuracy*100:.2f}%")
    
#     return model

#%%

#num_bits = 8**5 # Pour avoir un nombre de bits multiple de 8, Number of bits for each simulation
G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
k, _ = np.shape(G)
MAT = mat()
M = BPSK_modulation(mat_combi(G, k))

Eb_No_lin = 10**(1 / 10)  # Convert Eb/N0 values to linear scale # Eb_No_dB = 1  #Eb/N0 values in dB, on prend 1 pour les raisons évoqués dans le papier voir (a) polar code figure vert

combi_bits = mat().flatten()
encoded_bits = encodage_matrice(combi_bits, G, k)
X_train_flat = BPSK_modulation(encoded_bits) # Pas de buit ajouté pour l'instant
X_train = X_train_flat.reshape(256, 2*k)

# X_train = threshold_detection(X_train)

Y_train = result_comp(X_train_flat, G, MAT) # Evidament égale à "mat" mais nous vérfions tout de même
estimate_signals = np.array(Y_train).flatten()  # Flatten the array from (256, 16) to (1024,)

nb = 3*k
bits_neur = np.random.randint(0, 2, nb)
encoded_bits_test = encodage_matrice(bits_neur, G, k)
X_test_flat = BPSK_modulation(encoded_bits_test)
X_test = X_test_flat.reshape(nb // k, 2*k)

# detected_bits_test = threshold_detection(X_test)

Y_test = result_comp(X_test_flat, G, MAT)
estimate_signals = np.array(Y_test).flatten()

# ptit test
errors = count_errors(bits_neur, estimate_signals)
print(errors)  # normalement égale à bits_neur car pas de bruit

X_test = (X_test.T).astype(np.float64)

X_train = X_train.T
Y_train = Y_train.T
X_train = X_train.astype(np.float64)
Y_train = Y_train.astype(np.float64)
print('dimensions de X:', X_train.shape)
print('dimensions de y:', Y_train.shape)
#%%

training_history, parametres = deep_neural_network(X_train, Y_train, epochs=2**12, sigma=0.1)
#%%
Y_calcule = predict(X_test, parametres)
signal_recu = (Y_calcule.T).astype(int).flatten()
#%%


errors = count_errors(signal_recu, Y_test.flatten())

# Calculate BER
BER = errors/len(Y_test.flatten())

print("BER = ", BER, "\n", bits_neur,"\n", signal_recu)
