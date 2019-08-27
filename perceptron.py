"""
Rede Perceptron

@author Caio Tomich (caiotomich@gmail.com)
@version 1.0.0
@since 2019-08-27;
"""

import numpy as np
import matplotlib.pyplot as plt


def funcao_ativacao(val):
    if val < 0:
        return -1
    return 1


dados_treinamento = np.genfromtxt('documents/dados_treinamento.txt', skip_header=False)

t = np.full(30, -1)
w = np.random.uniform(low=-1, high=1, size=4)
x = np.hstack((t[:, np.newaxis], dados_treinamento[:, 0:3]))
d = dados_treinamento[:, 3]

print("\n --- Script de Treinamento -- \n")
print("Pesos Iniciais:", w)

epoca = 0
pesos = []
erro = True
taxa_aprendizagem = 0.01

while erro:
    erro = False

    for i in range(len(x)):
        u = np.dot(w, x[i])
        y = funcao_ativacao(u)

        if y != d[i]:
            w = w + (taxa_aprendizagem * (d[i] - y) * x[i])
            erro = True

    pesos.append(w)
    epoca += 1

print("Pesos Finais:", w)
print("Epocas:", epoca)

plt.plot(np.arange(epoca), np.array(pesos))
plt.show()


print("\n --- Script de Validação -- \n")

dados_validacao = np.genfromtxt('documents/dados_validacao.txt', skip_header=False)

wFinal = w
t = np.full(len(dados_validacao), -1)
x = np.hstack((t[:, np.newaxis], dados_validacao))
yFinal = []

for i in range(len(x)):
    u = np.dot(wFinal, x[i])
    y = funcao_ativacao(u)
    yFinal.append(y)

    if y == -1:
        print("Amostra {} pertence a classe P1".format(x[i]))
    else:
        print("Amostra {} pertence a classe P2".format(x[i]))

print("\nSaídas esperadas após o trinamento")
yEsperado = [-1, 1, 1, 1, 1, 1, -1, 1, -1, -1]
print("yEsperado == y: {}".format(np.equal(yEsperado, yFinal)))
