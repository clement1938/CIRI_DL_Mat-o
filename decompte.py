# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:13:35 2024

@author: 33695
"""

import time
from tqdm import tqdm
import pygame
import threading

# Initialiser pygame pour l'audio
pygame.init()
pygame.mixer.init()

# Chemin vers votre fichier audio
audio_path = 'son.mp3'

# Fonction pour jouer l'audio
def play_audio(audio_path, delay):
    time.sleep(delay)  # Attendre avant de jouer l'audio
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

# Durée estimée de la boucle en secondes
estimated_duration = 15  # Modifiez cette valeur selon vos mesures

# Démarrer la lecture de l'audio 10 secondes avant la fin de la boucle
delay = max(0, estimated_duration - 10)
audio_thread = threading.Thread(target=play_audio, args=(audio_path, delay))
audio_thread.start()

# Exécution de la boucle
start_time = time.time()
for i in tqdm(range(40000000)):
    a = (i**2 - (i-1)**2)
end_time = time.time()

# Affichage du temps d'exécution
print(f"La boucle a pris {end_time - start_time} secondes.")

# Attendre que l'audio ait fini de jouer avant de terminer le programme
audio_thread.join()

# Nettoyer pygame
pygame.quit()
