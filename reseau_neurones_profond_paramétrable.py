#%% Importation des librairies

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
import json                       # On téléchargera les BER dans des fichiers JSON  
import threading
from tqdm import tqdm             # Pour afiicher l'état d'avancement d'une boucle for
from sklearn.metrics import accuracy_score, log_loss

#%% Définition des fonctions
#%%% Partie notre réseau neuronal

def initialisation(dimensions):
    parametres = {}        # Initialisation du dictionnaire des paramètres
    C = len(dimensions)    # Nombre de couches
    np.random.seed(1)      # Initialisation de la graine pour la reproductibilité
    for c in range(1, C):  # On parcours les couches
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1]) * 0.01   # Initialisation des poids avec une distribution normale
        parametres['b' + str(c)] = np.zeros((dimensions[c], 1))  # Initialisation des biais à zéro
    return parametres    # Retourne le dictionnaire des paramètres

def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # Fonction d'activation sigmoïde

def ReLU(z):
    return np.maximum(0, z)      # Fonction d'activation ReLU

def forward_propagation(X, parametres):
    activations = {'A0': X}      # Initialisation des activations avec l'entrée
    C = len(parametres) // 2     # Nombre de couches
    for c in range(1, C):
        # Calcul de Z pour la couche c
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        # Application de la fonction ReLU
        activations['A' + str(c)] = ReLU(Z)
    # Calcul de Z pour la couche de sortie
    Z = parametres['W' + str(C)].dot(activations['A' + str(C - 1)]) + parametres['b' + str(C)]
    # Application de la fonction sigmoïde
    activations['A' + str(C)] = sigmoid(Z)
    return activations  # Retourne les activations de toutes les couches

def back_propagation(y, parametres, activations):
    m = y.shape[1]                      # Nombre d'exemples
    C = len(parametres) // 2            # Nombre de couches
    dZ = activations['A' + str(C)] - y  # Calcul de l'erreur pour la couche de sortie
    gradients = {}                      # Initialisation du dictionnaire des gradients

    # Gradient pour la couche de sortie (sigmoïde)
    gradients['dW' + str(C)] = 1/m * np.dot(dZ, activations['A' + str(C-1)].T)
    gradients['db' + str(C)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

    # Gradient pour les couches cachées (ReLU)
    for c in reversed(range(1, C)):
        dA_prev = np.dot(parametres['W' + str(c+1)].T, dZ)  # Gradient de l'activation précédente
        dZ = dA_prev * (activations['A' + str(c)] > 0)      # Dérivée de ReLU
        gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c-1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return gradients  # Retourne le dictionnaire des gradients

def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2  # Nombre de couches
    for c in range(1, C + 1):
        # Mise à jour des poids
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        # Mise à jour des biais
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]
    return parametres  # Retourne les paramètres mis à jour

def predict(X, parametres):
    activations = forward_propagation(X, parametres)  # Propagation avant pour obtenir les activations
    C = len(parametres) // 2        # Nombre de couches
    Af = activations['A' + str(C)]  # Activation finale
    return Af >= 0.5                # Retourne les prédictions (1 si >= 0.5, sinon 0)

def update_learning_rate(initial_lr, epoch, decay_rate):
    return initial_lr * np.exp(-decay_rate * epoch)  # Mise à jour du taux d'apprentissage avec un taux de décroissance pour opitmiser notre descente

def deep_neural_network(X, y, hidden_layers=(128, 64, 32), learning_rate=1, epochs=2**12, sigma=0, affichage=True):
    X = X.T.astype(np.float64)        # Conversion de X en type float64 et transposition
    y = y.T.astype(np.float64)        # Conversion de y en type float64 et transposition
    dimensions = list(hidden_layers)         # Liste des dimensions des couches cachées
    dimensions.insert(0, X.shape[0])         # Ajout de la dimension de l'entrée
    dimensions.append(y.shape[0])            # Ajout de la dimension de la sortie
    np.random.seed(1)                        # Initialisation de la graine pour la reproductibilité
    parametres = initialisation(dimensions)  # Initialisation des paramètres

    # Tableau numpy pour stocker l'historique de l'entraînement (log_loss et accuracy)
    training_history = np.zeros((int(epochs), 2))  # Initialisation pour pouvoir l'afficher plus tard
    C = len(parametres) // 2  # Nombre de couches
    m, n = X.shape

    # Boucle de descente de gradient
    for i in tqdm(range(epochs)):  # tqdm pour afficher l'état d'avancement des epochs
        lr = update_learning_rate(learning_rate, i, decay_rate=1/epochs)  # Mise à jour du taux d'apprentissage
        X_bruite = X + np.random.normal(0, sigma, (m, n))        # Ajout du bruit gaussien à chaque itération
        activations = forward_propagation(X_bruite, parametres)  # Propagation avant
        gradients = back_propagation(y, parametres, activations) # Rétropropagation
        parametres = update(gradients, parametres, lr)           # Mise à jour des paramètres
        Af = activations['A' + str(C)]                           # Activation finale

        # Calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
        y_pred = predict(X, parametres)
        training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    if affichage:
        # Plot courbe d'apprentissage
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(training_history[:, 0], label='train loss')  # Courbe du log_loss
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Courbe du Log Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(training_history[:, 1], label='train acc')   # Courbe de l'accuracy
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Courbe de l\'Accuracy')
        
        plt.tight_layout()
        plt.show()

    return training_history, parametres  # Retourne l'historique d'entraînement et les paramètres

def calculate_BER(parametres, X_test):
    X_test = (X_test.T).astype(np.float64)                   # Conversion de X_test en type float64 et transposition
    Y_calcule = predict(X_test, parametres)                  # Prédictions
    signal_recu = (Y_calcule.T).astype(int).flatten()        # Conversion des prédictions en type int et flatten
    errors = count_errors(signal_recu, Y_test.flatten())     # Comptage des erreurs de bits
    BER = float(errors / len(Y_test.flatten()))              # Calcul du BER
    print("BER = ", BER, "\n", bits_neur, "\n", signal_recu) # Affiche le BER et les vecteur initial et recu
    return BER  # Retourne le BER


#%%% Partie Telecom & librairie keras

def encodage_matrice(bits, G, k):
    bits_reshaped = bits.reshape(len(bits)//k, k)  # On reshape des bits en blocs de taille k
    bits_encodes = bits_reshaped @ G % 2           # Encodage des bits en utilisant la matrice G, définie plus tard
    return bits_encodes.flatten()                  # Flatten le résultat pour obtenir un tableau 1D

def mat_combi(G, k):
    n_combinations = 2**k                          # Calcul du nombre de combinaisons possibles
    matrice_combinaisons = np.zeros((n_combinations, k), dtype=int)  # Initialisation de la matrice de combinaisons
    for i in range(n_combinations):                # On remplie la matrice avec des combinaisons de 8 bits
        binary_string = format(i, '08b')           # Et puis on convertit l'entier i en une chaîne binaire de 8 caractères
        bit_array = np.array([int(bit) for bit in binary_string])  # Conversion de la chaîne binaire en tableau de bits
        matrice_combinaisons[i, :] = bit_array     # Ajout de la combinaison à la matrice
    return (matrice_combinaisons@G)%2  # Return la matrice combinée avec G

def mat():
    n_combinations = 2**k                 # Calcul du nombre de combinaisons possibles
    matrice_combinaisons = np.zeros((n_combinations, k), dtype=int)  # Initialisation de la matrice de combinaisons
    for i in range(n_combinations):       # On remplie la matrice avec des combinaisons de 8 bits
        binary_string = format(i, '08b')  # Conversion l'entier i en une chaîne binaire de 8 caractères
        bit_array = np.array([int(bit) for bit in binary_string])  # On fait pareil, on convertit la chaîne binaire en tableau de bits
        matrice_combinaisons[i, :] = bit_array  # Ajouter la combinaison à la matrice
    return matrice_combinaisons           # Et return la matrice de combinaisons

def BPSK_modulation(bits):
    return (2*bits - 1).astype(int)       # Modulation BPSK des bits (0 -> -1, 1 -> 1)

def threshold_detection(received_signals):
    return received_signals > 0           # Détection de seuil pour la démodulation BPSK    

def count_errors(original_bits, detected_bits):
    return np.sum(original_bits != detected_bits)  # Compter le nombre d'erreurs de bits

# Theoretical BER calculation using Q-function
def theoretical_ber(Eb_No):
    return 0.5 * sp.erfc(np.sqrt(Eb_No))   # On calcule du BER théorique en utilisant la fonction Q

def result_comp(received_signals, G, MAT):
    lignes_correspondante = []                     # Et puis l'initialisation de la liste des lignes correspondantes
    for i in range(len(received_signals)//(2*k)):  # Parcours des signaux reÃ§us par blocs de taille 2*k
        distances = np.sum(np.abs(M - received_signals[16*i:16*(i+1)]), axis=1)  # On calcule des distances avec les lignes de M
        indice_min = np.argmin(distances)          # Trouver l'indice de la ligne avec la distance minimale
        ligne_correspondante = MAT[indice_min, :]  # Récupérer la ligne correspondante de la matrice_combinaisons
        lignes_correspondante.append(ligne_correspondante)  # Et puis ajouter la ligne correspondante à la liste
    return np.array(lignes_correspondante)         # On finit par return les lignes correspondantes estimées

# On vérifie que la bibliothèque moviepy est installée (pour l'audio)
# Précision, on met en place cet audio pour prévenir de la fin imminente du programme, ne pas mettre le son trop fort !
try :
    from moviepy.editor import AudioFileClip
    audio_installed = True
except ImportError:
    audio_installed = False

# On crée la fonction pour jouer l'audio
def play_audio(audio_path, delay):
    if audio_installed:
        audio_clip = AudioFileClip(audio_path)
        audio_clip.preview()
    else:
        print("La bibliothèque moviepy n'est pas installée. L'audio ne sera pas joué.")

@tf.autograph.experimental.do_not_convert  # Pour éviter un bugg de type
def ber_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.cast(tf.not_equal(y_true, tf.round(y_pred)), tf.float32))  # Calcul de la métrique BER

def add_noise(data, noise_level):
    noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=noise_level)        # Génération du bruit normal
    return data + noise          # Ajout du bruit aux données d'entrée

def train_neural_network(X_train, Y_train, X_test, Y_test, noise_level, epochs):
    inputs = Input(shape=(16,))  # On définit les entrées du modèle
    noisy_inputs = Lambda(add_noise, arguments={'noise_level': noise_level})(inputs)    # Ajout de bruit aux entrées

    # Et puis, on construit le réseau
    x = Dense(128, activation='relu')(noisy_inputs)  # Première couche dense avec activation ReLU
    x = Dense(64, activation='relu')(x)              # Deuxième couche dense avec activation ReLU
    x = Dense(32, activation='relu')(x)              # Troisième couche dense avec activation ReLU
    outputs = Dense(8, activation='sigmoid')(x)      # Couche de sortie avec activation sigmoide

    # Création du modèle
    model = Model(inputs=inputs, outputs=outputs)  

    # On compile le modèle avec l'optimiseur Adam, la perte MSE et la métrique BER
    model.compile(optimizer='adam', loss='mse', metrics=[ber_metric])  

    # On entraîne le modèle avec validation
    model.fit(X_train, Y_train, epochs=epochs, verbose=0)  # Entraînement du modèle

    # Et on finit par l'évaluer ! (sur les données de test)
    loss, ber = model.evaluate(X_test, Y_test)  
    print(f"BER: {ber:.5f}")  # On temrine avec l'affichage du BER
    
    return model, ber


#%% Définition des variables

G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Matrice génératice, donnée dans le TP2
              [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  

k, _ = np.shape(G)    # On extrait le nombre de lignes de G, donc la taille de la sortie
MAT = mat()           # Génération de la matrice de combinaisons
M = BPSK_modulation(mat_combi(G, k))  # Modulation de la matrice de combinaison

Eb_No_lin = 10**(1 / 10)    # On converit les valeurs Eb/N0 en échelle linéaire
std = np.sqrt(1/Eb_No_lin)  # Et on calcule l'écart type pour le bruit

#%% Test modulation/démodulation et définition de nouvelles variables

combi_bits = mat().flatten()                       # On génère et aplatis les bits de combinaisons
encoded_bits = encodage_matrice(combi_bits, G, k)  # Encodage des bits de combinaisons
X_train_flat = BPSK_modulation(encoded_bits)       # On module les bits encodés sans bruit ajouté
X_train = X_train_flat.reshape(256, 2*k)           # Et reshape des bits modulés en un tableau 2D

Y_train = result_comp(X_train_flat, G, MAT)        # Calcul des résultats correspondants pour les training
estimate_signals = np.array(Y_train).flatten()     # On flatten les signaux estimés (pour avoir la bonne dimension)

nb = 100000*k                                          # On définit le nombre de bits
bits_neur = np.random.randint(0, 2, nb)                # Génère des bits aléatoires
encoded_bits_test = encodage_matrice(bits_neur, G, k)  # Et encode les bits aléatoires
X_test_flat = BPSK_modulation(encoded_bits_test)       # Et puis module les bits encodés
X_test = X_test_flat.reshape(nb // k, 2*k)             # Finalement, reshape les bits modulés en un tableau 2D

Y_test = result_comp(X_test_flat, G, MAT)        # Calcul des résultats correspondants pour les tests
estimate_signals = np.array(Y_test).flatten()    # Et on fait pareil

# Test de bon fonctionnement
errors = count_errors(bits_neur, estimate_signals)  # Compte les erreurs de bits
if errors == 0:
    # On vérifie si la modulation et la démodulation fonctionnent correctement sans bruit
    print("La modulation et démodulation fonctionnent correctement")         
else:
    # Et affichage d'un message d'erreur si la modulation et la démodulation ne fonctionnent pas 
    print("La modulation et démodulation ne fonctionnent pas correctement")  

#%% Test du réseau de neurones de keras
# On fait un test pour Eb/N0 = 1 & epochs=2**12  (environ une minute) 

start_time_keras = time.time()
trained_model, BER = train_neural_network(X_train, Y_train, X_test, Y_test, noise_level = std, epochs=2**12)
end_time_keras = time.time()
execution_time_keras = end_time_keras - start_time_keras
print(f"Temps d'exécution pour le réseau de neurones Keras : {execution_time_keras:.2f} secondes")

#%% Test de notre réseau de neurones

start_time_notre = time.time()
training_history, parametres = deep_neural_network(X_train, Y_train, epochs=2**12, sigma=std)
BER = calculate_BER(parametres, X_test)
end_time_notre = time.time()
execution_time_notre = end_time_notre - start_time_notre
print(f"Temps d'exécution pour notre propre réseau de neurones : {execution_time_notre:.2f} secondes")

#%% Définition du main

# Création d'une fonction pour faire une courbe (Mep = 2^a) le graphique (a) de l'étude présentée

def main(epochs, a, reseau='keras'):
    Eb_N0 = [i for i in range(0,11)]  # Definition des valeurs de Eb/N0
    BER = []                          # Init de BER
    for i in tqdm(Eb_N0):             # Boucle sur les valeurs de Eb/N0 avec affichage de progression (important !)
        std = np.sqrt(10**(-i/10))    # On compute l'écart type pour le bruit
        print("\n", "etape ", i, "/10", "\n", "std =", std, "\n")  # On affiche là ou nous en sommes
        
        # Et a ce moement, on peut directement entraîner le modèle et obteneir le BER
        if reseau == 'keras':
            trained_model, BER_i = train_neural_network(X_train, Y_train, X_test, Y_test, noise_level=std, epochs=epochs)  
        elif reseau == 'notre':
            training_history, parametres = deep_neural_network(X_train, Y_train, epochs=epochs, sigma=std, affichage=False)
            BER_i = calculate_BER(parametres, X_test)
        else:
            print("le modèle de réseau neuronal n'existe pas")
        BER.append(BER_i)   # On append le nouveau BER à la liste
        
        try:                                # Mis en place de l'audio
            if i == 9 and audio_installed:  # Afin de prévenir que le code est sur le point de finir de s'exécuter
                audio_thread = threading.Thread(target=play_audio, args=('son.mp3', 1))
                audio_thread.start()        # Il prend place à la 10e itération 
        except Exception as e:
            print(f"Erreur lors de la lecture de l'audio : {e}")

    plt.semilogy(Eb_N0, BER, marker='o', linestyle='-')  # Tracé du BER en échelle semi-logarithmique
    plt.xlabel('Eb/N0 (dB)')                          
    plt.ylabel('BER')                                    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show() 

    with open(f'ber_{reseau}_list_{a}.json', 'w') as f:  # Enregistre les résultats de BER dans un fichier JSON pour ne pas perdre les résultats
        json.dump(BER, f)                       # Sauvegarde les BER
                                                # De cette manière, pas besoin de refaire les calculs plusieurs fois
    return BER  # Retourne la liste des BER

#%%  Appel du main

a=10   # 2^a, le nombre d'epochs

# On a caché la ligne ci-dessous caché car le temps de calcul est élevé, vous pouvez les run en les décommentant

#BER = main(2**a, a, reseau='keras')  # Appelle la fonction main pour exécuter le programme principal (réseau keras)
                                     # On peut choisir quel réseau de neurones on veut uiliser
#BER = main(2**a, a, reseau='notre')  # Appelle la fonction main avec notre réseau


#%% Affichage de toutes les courbes

with open('ber_keras_list_10.json', 'r') as f:  # On ouvre les fichier Json contenant
    BER10 = json.load(f)                        # les BER en fonction des Eb/No
with open('ber_keras_list_12.json', 'r') as f:
    BER12 = json.load(f)
with open('ber_keras_list_14.json', 'r') as f:
    BER14 = json.load(f)
with open('ber_keras_list_16.json', 'r') as f:
    BER16 = json.load(f)
with open('ber_keras_list_18.json', 'r') as f:
    BER18 = json.load(f)

Eb_N0 = [i for i in range(0,11)]                             # Définis les valeurs de Eb/N0
ber_theorique = [theoretical_ber(10**(i/10)) for i in Eb_N0] # On prend les ber théoriques pour les plot sur le même graph

# Affichage des courbes avec légendes
plt.semilogy(Eb_N0, BER10, color='g', marker='o', linestyle='-', label='M_ep = 2^10')
plt.semilogy(Eb_N0, BER12, color='b', marker='o', linestyle='-', label='M_ep = 2^12')
plt.semilogy(Eb_N0, BER14, color='orange', marker='o', linestyle='-', label='M_ep = 2^14')
plt.semilogy(Eb_N0, BER16, color='r', marker='o', linestyle='-', label='M_ep = 2^16')
plt.semilogy(Eb_N0, BER18, color='purple', marker='o', linestyle='-', label='M_ep = 2^18')
plt.semilogy(Eb_N0, ber_theorique, color='k', marker='x', linestyle='--', label='BER Théorique') # On trace la courbe théoriqueen parallèle
plt.xlabel('Eb/N0 (dB)')  # On ajoute un label à  l'axe des abscisses
plt.ylabel('BER')  
plt.grid(True, which='both', linestyle='--', linewidth=0.5) # On affiche le fond
plt.legend()
plt.title('Courbes BER')
plt.show()
