import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Lambda

#%%

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


# def add_noise(signals, sigma):
#     noise = np.random.normal(0, sigma, signals.shape)
#     return signals + noise


def train_neural_network(X_train, Y_train, X_test, Y_test, noise_level, epochs=2**10, learning_rate=0.5):
    model = Sequential([
        # Ajouter du bruit aux données d'entrée
        Lambda(lambda x: tf.cast(x, tf.float32) + tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_level)),
        Dense(16, input_dim=X_train.shape[1], activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(Y_train.shape[1], activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entraîner le modèle
    model.fit(X_train, Y_train, epochs=epochs, verbose=0)

    # Évaluer le modèle
    _, accuracy = model.evaluate(X_test, Y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    return model

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # # Add noise to training data
    # X_train_noisy = add_noise(X_train, noise_level)  # A changer car le même pour ttes les epochs

    # Train the model
    model.fit(X_train, Y_train, epochs=epochs, verbose=0)

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, Y_test)
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    return model

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

#%%

combi_bits = mat().flatten()
encoded_bits = encodage_matrice(combi_bits, G, k)
X_train_flat = BPSK_modulation(encoded_bits) # Pas de buit ajouté pour l'instant
X_train = X_train_flat.reshape(256, 2*k)

# X_train = threshold_detection(X_train)

Y_train = result_comp(X_train_flat, G, MAT) # Evidament égale à "mat" mais nous vérfions tout de même
estimate_signals = np.array(Y_train).flatten()  # Flatten the array from (256, 16) to (1024,)

#%%

nb = 80*k
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

trained_model = train_neural_network(X_train, Y_train, X_test, Y_test, noise_level = 0.0089, epochs=2**12)
prediction = trained_model.predict(X_test)
signal_recu = (prediction > 0.5).flatten().astype(int)
# Count errors
errors = count_errors(Y_test.flatten(), signal_recu)

# Calculate BER
BER = errors/len(Y_test.flatten())

print("BER = ", BER) #, "\n", bits_neur,"\n", signal_recu)


#%%

# Generate random bits
# bits = np.random.randint(0, 2, num_bits)
# encoded_bits = encodage_matrice(bits, G, k)    
# # BPSK Modulation/mapping
# transmitted_signals = BPSK_modulation(encoded_bits)

# # AWGN Channel/add noise
# sigma = np.sqrt(10**(-Eb_No_lin / 10))                           # A peaufiner
# noise = np.random.normal(0, sigma, len(transmitted_signals))
# received_signals = transmitted_signals + noise

# # Threshold detection
# detected_bits = threshold_detection(received_signals)
# estimate_signals = result_comp(received_signals, G) # On garde ça pour comparer si jamais
# estimate_signals = np.array(estimate_signals).flatten()  # Flatten the array from (64, 16) to (1024,)
# estimate_signals2 = train_neural_network(received_signals)

# # Count errors
# errors = count_errors(transmitted_signals, estimate_signals)

# # Calculate BER
# BER = errors / (2*num_bits)
# simulated_BER.append(BER)

# # Theoretical BER calculation
# theoretical_BER = theoretical_ber(Eb_No_lin)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.semilogy(Eb_No_dB, simulated_BER, 'o-', label='Simulated BER')
# plt.semilogy(Eb_No_dB, theoretical_BER, 'r--', label='Theoretical BER')
# plt.xlabel('Eb/N0 (dB)')
# plt.ylabel('Bit Error Rate (BER)')
# plt.title('BER vs. Eb/N0 for BPSK in AWGN')
# plt.legend()
# plt.grid(True)
# plt.show()
