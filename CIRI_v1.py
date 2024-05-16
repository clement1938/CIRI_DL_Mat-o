import numpy as np
import scipy.special as sp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
import json
import threading
from moviepy.editor import *
import matplotlib.pyplot as plt

#%% Définition des libraires
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
    n_combinations = 2**k
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
def threshold_detection(received_signals):
    return received_signals > 0           

# Counting the number of bit errors
def count_errors(original_bits, detected_bits):
    return np.sum(original_bits != detected_bits)

# Theoretical BER calculation using Q-function
def theoretical_ber(Eb_No):
    return 0.5 * sp.erfc(np.sqrt(Eb_No))

def result_comp(received_signals, G, MAT):
    lignes_correspondante = []
    for i in range(len(received_signals)//(2*k)):
        distances = np.sum(np.abs(M - received_signals[16*i:16*(i+1)]), axis=1)
        indice_min = np.argmin(distances) # Trouve l'indice de la ligne avec la distance minimale (max de vraisemblance)
        ligne_correspondante = MAT[indice_min, :] # Récupère la ligne correspondante de la matrice_combinaisons
        lignes_correspondante.append(ligne_correspondante)
    return np.array(lignes_correspondante) # estimé

def play_audio(audio_path, delay):
    audio_clip = AudioFileClip(audio_path)
    audio_clip.preview()

# Définir la fonction de métrique BER
@tf.autograph.experimental.do_not_convert
def ber_metric(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.not_equal(y_true, tf.round(y_pred)), tf.float32))

# Fonction pour ajouter du bruit
def add_noise(data, noise_level):
    noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=noise_level)
    return data + noise

# Fonction pour entraîner le réseau neuronal
def train_neural_network(X_train, Y_train, X_test, Y_test, noise_level, epochs):
    inputs = Input(shape=(16,))
    # Ajout de bruit
    noisy_inputs = Lambda(add_noise, arguments={'noise_level': noise_level})(inputs)

    # Construction du réseau
    x = Dense(128, activation='relu')(noisy_inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(8, activation='sigmoid')(x)

    # Création du modèle
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse', metrics=[ber_metric])

    # Entraîner le modèle avec validation
    model.fit(X_train, Y_train, epochs=epochs, verbose=0)

    # Évaluation du modèle
    loss, ber = model.evaluate(X_test, Y_test)
    print(f"BER: {ber:.5f}")
    
    return model


#%% Définition des variables
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
std = np.sqrt(1/Eb_No_lin)

#%% Test

combi_bits = mat().flatten()
encoded_bits = encodage_matrice(combi_bits, G, k)
X_train_flat = BPSK_modulation(encoded_bits) # Pas de buit ajouté pour l'instant
X_train = X_train_flat.reshape(256, 2*k)

Y_train = result_comp(X_train_flat, G, MAT) # Evidemment égale à "mat" mais nous vérfions tout de même
estimate_signals = np.array(Y_train).flatten()  # Flatten the array from (256, 16) to (1024,)

nb = 1000*k
bits_neur = np.random.randint(0, 2, nb)
encoded_bits_test = encodage_matrice(bits_neur, G, k)
X_test_flat = BPSK_modulation(encoded_bits_test)
X_test = X_test_flat.reshape(nb // k, 2*k)

Y_test = result_comp(X_test_flat, G, MAT)
estimate_signals = np.array(Y_test).flatten()

# test de bon fonctionnement
errors = count_errors(bits_neur, estimate_signals)
if errors == 0:
    print("La modulation et démodulation fonctionneent correctement")  # normalement égale à bits_neur car pas de bruit
else:
    print("La modulation et démodulation ne fonctionneent pas correctement")  # normalement égale à bits_neur car pas de bruit

#%% Training

trained_model = train_neural_network(X_train, Y_train, X_test, Y_test, noise_level = std, epochs=2**10)
prediction = trained_model.predict(X_test)
signal_recu = (prediction > 0.5).flatten().astype(int)
# Count errors
errors = count_errors(Y_test.flatten(), signal_recu)

# Calculate BER
BER = errors/len(Y_test.flatten())

print("BER = ", BER) #, "\n", bits_neur,"\n", signal_recu)


#%%

def main (epochs, a):
    Eb_N0 = [i for i in range(0,11)]
    BER = []
    for i in Eb_N0 :
        std = np.sqrt(10**(-i/10))
        # Boucle sur epochs
        trained_model = train_neural_network(X_train, Y_train, X_test, Y_test, noise_level = std, epochs=epochs)
        # _ , parametres = deep_neural_network(X_train, Y_train, epochs=epochs, sigma=sigma, affichage = False)
        # errors = count_errors(predict(X_test, parametres), Y_test.flatten())
        BER_i = errors/(len(Y_test.flatten())*10**(a/2))
        BER.append(BER_i)
        print("BER ", i, " = ", BER_i, "\n")
    audio_path = 'son.mp3'
    audio_thread = threading.Thread(target=play_audio, args=(audio_path, 1))
    audio_thread.start()
    plt.semilogy(Eb_N0, BER, marker='o', linestyle='-')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Et epoch
    with open(f'ber_list_{a}.json', 'w') as f:
        json.dump(BER, f)
    
    return BER

#%%

a=8
BER = main(2**a, a)

#%% Autre

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
