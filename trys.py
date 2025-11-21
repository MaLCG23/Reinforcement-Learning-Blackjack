import random
from collections import deque
import os
import pickle
import numpy as np
from scipy.stats import norm

class mesa:
    def __init__(self):
        self.mazo = deque()
        self.mano_jugador = []
        self.mano_banca = []
        self.apostado = False

    def reset(self):
        lista_base = [i for i in range(2, 15)] * 4
        random.shuffle(lista_base)
        self.mazo = deque(lista_base)
        self.mano_banca = [self.mazo.popleft()]
        self.mano_jugador = [self.mazo.popleft(), self.mazo.popleft()]
        self.apostado = False

    def pedir_carta_jugador(self):
        self.mano_jugador.append(self.mazo.popleft())

    def pedir_carta_banca(self):
        self.mano_banca.append(self.mazo.popleft())

    def apostar(self):
        self.apostado = True

    def calcular_puntos(self, mano):
        totales = [0]
        for carta in mano:
            if 11 <= carta <= 13:  # J, Q, K
                totales = [t + 10 for t in totales]
            elif carta == 14:  # As
                nuevos = []
                for t in totales:
                    nuevos.append(t + 1)
                    nuevos.append(t + 11)
                totales = nuevos
            else:
                totales = [t + carta for t in totales]
        return sorted(set(totales))

    def total_estado(self, mano):
        puntos = self.calcular_puntos(mano)
        validos = [p for p in puntos if p <= 21]
        return max(validos) if validos else min(puntos)

    def resolver(self):
        if len(self.mano_banca) < 2:
            self.mano_banca.append(self.mazo.popleft())

        puntos_jugador = self.total_estado(self.mano_jugador)
        puntos_banca = self.total_estado(self.mano_banca)

        if puntos_jugador > 21:
            return -2 if self.apostado else -1
        while self.total_estado(self.mano_banca) < 17:
            self.pedir_carta_banca()
        puntos_banca = self.total_estado(self.mano_banca)

        if puntos_banca > 21:
            return 2 if self.apostado else 1
        if puntos_jugador > puntos_banca:
            return 2 if self.apostado else 1
        elif puntos_jugador < puntos_banca:
            return -2 if self.apostado else -1
        else:
            return 0

    def get_state(self):
        puntos_jugador = self.calcular_puntos(self.mano_jugador)
        estado_puntos_jugador = self.total_estado(self.mano_jugador)
        as_jugador = len(puntos_jugador) - 1

        return (self.mano_banca[0], estado_puntos_jugador, as_jugador, len(self.mano_jugador), self.apostado)


# Q-learning base
def get_Q(Q, estado, accion): return Q.get((estado, accion), 0.0)

def elegir_accion(Q, estado, acciones):
    valores = [get_Q(Q, estado, a) for a in acciones]
    maximo = max(valores)
    mejores = [a for a, v in zip(acciones, valores) if v == maximo]
    return random.choice(mejores)


acciones = [0, 1, 2]
Q = {}
if os.path.exists("C:/FBI/q_table.pkl"):
    with open("C:/FBI/q_table.pkl", "rb") as f:
        Q = pickle.load(f)

actualizaciones = 1000000
historiala = np.zeros(actualizaciones)
historialb = np.zeros(actualizaciones)


for i in range(actualizaciones):
    meseta = mesa()
    meseta.reset()
    estado = meseta.get_state()
    has_jugado = False
    recompensa = 0

    while True:
        accion = elegir_accion(Q, estado, acciones)

        if accion == 0:
            meseta.pedir_carta_jugador()
            if meseta.total_estado(meseta.mano_jugador) > 21:
                recompensa = -1
                nuevo_estado = meseta.get_state()
                break
            recompensa = 0
            nuevo_estado = meseta.get_state()

        elif accion == 1:
            recompensa = meseta.resolver()
            nuevo_estado = meseta.get_state()
            break

        elif accion == 2 and not has_jugado:
            meseta.apostar()
            recompensa = 0
            nuevo_estado = meseta.get_state()
        else:
            recompensa = -2
            nuevo_estado = meseta.get_state()
            break
        estado = nuevo_estado
        has_jugado = True
    historiala[i] = recompensa

for i in range(actualizaciones):
    meseta = mesa()
    meseta.reset()
    estado = meseta.get_state()
    has_jugado = False
    recompensa = 0

    while True:
        accion = random.choice(acciones)

        if accion == 0:
            meseta.pedir_carta_jugador()
            if meseta.total_estado(meseta.mano_jugador) > 21:
                recompensa = -1
                nuevo_estado = meseta.get_state()
                break
            recompensa = 0
            nuevo_estado = meseta.get_state()

        elif accion == 1:
            recompensa = meseta.resolver()
            nuevo_estado = meseta.get_state()
            break

        elif accion == 2 and not has_jugado:
            meseta.apostar()
            recompensa = 0
            nuevo_estado = meseta.get_state()
        else:
            recompensa = -2
            nuevo_estado = meseta.get_state()
            break
        estado = nuevo_estado
        has_jugado = True
    historialb[i] = recompensa







#COMPROBATION CODE:
#This code is made to compare the performance of an IA vs a random agent in a blackjack

#Here I'll show if my code works (if it does) by comparison on how much good/bad is form the
#same code runed randomlly. This data is made with 25833 * 1000 = 25833000 Q actualizations, 
#but i let the code so anyone can reproduce the results or change the number of samples
mean_ia = np.mean(historiala)
var_ia = np.var(historiala)

mean_random = np.mean(historialb)
var_random = np.var(historialb)

#With this data we can aplly the central limit theorem to see how good is the IA vs random
#H0: mean_ia == mean_random
#H1: mean_ia > mean_random
#We use Z test for 2 independent samples: 
# Z = (mean_ia - mean_random)-delta/ sqrt(var_ia/N + var_random/N)
# where delta = 0 under H0
z = ((mean_ia - mean_random)-(0.75*mean_random)) / np.sqrt(var_ia/actualizaciones + var_random/actualizaciones)

#Now we use the unilateral test to see the p-value, with a alpha = 0.05 (95% confidence)
p_value = 1 - norm.cdf(z)
if p_value < 0.05:
    print("ia plays better than random with a confidence of 95%")
else:
    print("we dont know if ia play better than random with a confidence of 95%")