# -*- coding: utf-8 -*-
"""
Created on Fri May 12 23:51:14 2023

@author: diego
"""

import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt


def matriz_to_vector(matriz):
    # toma una matriz de adyacencia nxn
    # devuelve un vector con los componentes de la triangular superior
    # acomodados fila por fila
    
    n = matriz.shape[0]
    # obtener los indices superiores (sin la diagonal)
    upper_indices = np.triu_indices(n, k=1)
    # obtener estos valores de la matriz
    vector = matriz[upper_indices]

    return vector

def vector_to_matriz(vector):
    # toma un vector de m elementos
    # los acomoda fila por fila en la triangular superior
    # para formar la matriz de adyacencia asociada
    
    m = len(vector)
    # recordando la formula
    n = (1+math.sqrt(1+8*m))/2
    # comprobar que n si es un numero natural
    assert n.is_integer() and  n >= 1
    # hacer que entonces ya se tome como entero
    n = int(n)
    # crear la matriz nxn y llenarla de 0's
    matriz = np.zeros((n, n))
    # Hacer que el traignulo superior de la matriz (sin la diagonal) sea nuestro vector
    upper_indices = np.triu_indices(n, k=1)  # Obtener los índices superiores de la matriz
    matriz[upper_indices] = vector
    # Hacer que el traingulo inferior de la matriz (sin la diagonal) tambien sea el vector
    matriz = matriz + matriz.T
    
    return matriz

def matriz_to_lista(matriz):
    # toma una matriz de adyacencia nxn
    # devuelve la lista de adyacencia (lista con n elementos (listas), denotando conexiones)
    # si i, j son vertcies (numeros de 0 a n-1) con i>j y (i, j) es una artista entonces
    # el elemento i de la lista contiene a j, pero el elemento j de la lista no contiene a i
    # para asi evitar redundancia de informacion
    
    n = matriz.shape[0]
    lista = [] # lista de adyacencia
    
    # iterar en los nodos 0, 1, ..., n-1
    for i in range(n):
        conexiones_i = []
        # iterar en los nodos 0, 1, ..., i-1
        for j in range(i):
            # si hay conexion (i, j) se agrega j a las conexiones de i
            if matriz[i, j] == 1:
                conexiones_i += [j]
        # agregar las conexiones de i a la lista
        lista += [conexiones_i]
    
    return lista

def lista_to_matriz(lista):
    # toma una lista de adyacencia (de n elemetnos), donde:
    # si i, j son vertcies (numeros de 0 a n-1) con i>j y (i, j) es una artista entonces
    # el elemento i de la lista contiene a j
    # devuelve la matriz de adyaciencia
    
    n = len(lista)
    matriz = np.zeros((n, n)) # primero no hay conexiones
    
    # iterar en el indice del vertice y sus conexiones
    for i, conexiones_i in enumerate(lista):
        # agregar un 1 en las conexiones que tenga el vertice i
        matriz[i, conexiones_i] = 1
    # recordar que es simetrica
    matriz = matriz + matriz.T
        
    return matriz

def lista_to_vector(lista):
    # toma una lista de adyacencia (de n elemetnos), donde:
    # si i, j son vertcies (numeros de 0 a n-1) con i>=j y (i, j) es una artista entonces
    # el elemento i de la lista contiene a j
    # devuelve un vector con los componentes de la triangular superior de la mattriz de adyacencia
    # acomodados fila por fila
    
    matriz_adyacencia = lista_to_matriz(lista)
    vector = matriz_to_vector(matriz_adyacencia)
    
    return vector

def vector_to_lista(vector):
    # toma un vector de m elementos que son los componentes de la triangular superior
    # de la mattriz de adyacencia acomodados fila por fila
    # devuelve una lista de adyacencia (de n elemetnos), donde:
    # si i, j son vertcies (numeros de 0 a n-1) con i>=j y (i, j) es una artista entonces
    # el elemento i de la lista contiene a j
    
    matriz_adyacencia = vector_to_matriz(vector)
    lista = matriz_to_lista(matriz_adyacencia)
    
    return lista

def matriz_to_edge_index(matriz):
    # toma una matriz de adyacencia nxn
    # devuelve un arreglo de tipo edge_index, es decir
    # es un arreglo de tamaño [2, num_edges] donde
    # [0, idx] = i, [1, idx] = j, significa que los nodos i y j estan conectados
    
    return np.array(np.nonzero(matriz))


def edge_index_to_matriz(edge_index, n=0):
    # toma un array de tipo edge_index, es decir
    # un arreglo de tamaño [2, num_edges] donde
    # [0, idx] = i, [1, idx] = j, significa que los nodos i y j estan conectados
    # devuelve la matriz de adyacencia del grafo
    # devuelve la matriz de adyacencia
    
    # n - numero de nodos en el grafo
    # si este numero no se da, se asume que es el nodo mas grande conectado
    if n == 0:
        n = edge_index.max() + 1
    
    matriz = np.zeros((n, n)) # poner zeros
    
    # poner 1 donde hay aristas
    for index in edge_index.T:
    # iterar en pares [i, j] que representan aristas
        matriz[index[0], index[1]] = 1
    
    return matriz
    

def vector_to_edge_index(vector):
    # toma un vector de m elementos que son los componentes de la triangular superior
    # devuelve un arreglo de tipo edge_index, es decir
    # es un arreglo de tamaño [2, num_edges] donde
    # [0, idx] = i, [1, idx] = j, significa que los nodos i y j estan conectados
    
    matriz = vector_to_matriz(vector)
    edge_index = matriz_to_edge_index(matriz)
    
    return edge_index


def edge_index_to_vector(edge_index):
    # toma un arreglo de tipo edge_index, es decir
    # es un arreglo de tamaño [2, num_edges] donde
    # [0, idx] = i, [1, idx] = j, significa que los nodos i y j estan conectados
    # devuelve un vector de m elementos que son los componentes de la triangular superior
    
    matriz = edge_index_to_matriz(edge_index)
    vector = matriz_to_vector(matriz)
    return vector

def lista_to_edge_index(lista):
    # toma una lista de adyacencia (de n elemetnos), donde:
    # si i, j son vertcies (numeros de 0 a n-1) con i>=j y (i, j) es una artista entonces
    # el elemento i de la lista contiene a j
    # devuelve un arreglo de tipo edge_index, es decir
    # es un arreglo de tamaño [2, num_edges] donde
    # [0, idx] = i, [1, idx] = j, significa que los nodos i y j estan conectados
    
    matriz = lista_to_matriz(lista)
    edge_index = matriz_to_edge_index(matriz)
    
    return edge_index

def edge_index_to_lista(edge_index):
    # toma un arreglo de tipo edge_index, es decir
    # es un arreglo de tamaño [2, num_edges] donde
    # [0, idx] = i, [1, idx] = j, significa que los nodos i y j estan conectados
    # devuelve una lista de adyacencia (de n elemetnos), donde:
    # si i, j son vertcies (numeros de 0 a n-1) con i>=j y (i, j) es una artista entonces
    # el elemento i de la lista contiene a j
    
    matriz = edge_index_to_matriz(edge_index)
    lista = matriz_to_lista(matriz)
    
    return lista
    
    
def dibujar_grafo_circular_from_matriz(matriz_adyacencia, ax=None, titulo=None):
    # ax puede ser el eje donde se quiere graficar
    # si no se pasa un ax, entonces se grafica y se muestra
    # tambien se le puede pasar titulo
        
    # Crear un objeto de grafo
    G = nx.from_numpy_matrix(matriz_adyacencia)

    # Obtener el número de nodos
    num_nodos = G.number_of_nodes()

    # Crear un diseño de posición spring
    pos = nx.circular_layout(G)
    
    ver = 0 # no se muestra la figura
    if ax is None:
        ver = 1 # al menos que no se proporcione ax
        # Crear una nueva figura y ejes si no se proporciona ax
        fig, ax = plt.subplots(figsize=(5, 5))

    # Dibujar los nodos
    tamaño_nodos = min(500 /((num_nodos)**1/10), 500) # mientras mas nodos, mas pequeños
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size= tamaño_nodos, ax=ax)
    
    # Dibujar las aristas
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)

    # Dibujar las etiquetas de los nodos
    tamaño_etiqueta = min(10 /((num_nodos)**1/20), 10) # mientras mas nodos, mas pequeños
    nx.draw_networkx_labels(G, pos, font_size= tamaño_etiqueta, font_color='black', ax=ax)
    
    ax.axis('off')
    if titulo:
        ax.set_title(titulo)

    if ver == 1:
        # Mostrar la figura
        plt.show()
        
def dibujar_grafo_circular_from_vector(vector, ax=None, titulo=None):
    
    matriz_adyacencia = vector_to_matriz(vector)
    
    dibujar_grafo_circular_from_matriz(matriz_adyacencia, ax, titulo)


def dibujar_grafo_circular_from_lista(lista, ax=None, titulo=None):
    
    matriz_adyacencia = lista_to_matriz(lista)
    
    dibujar_grafo_circular_from_matriz(matriz_adyacencia, ax, titulo)
    
def dibujar_grafo_circular_from_edge_index(edge_index, ax=None, titulo=None):
    
    matriz_adyacencia = edge_index_to_matriz(edge_index)
    
    dibujar_grafo_circular_from_matriz(matriz_adyacencia, ax, titulo)


def dibujar_grafo_from_matriz(matriz_adyacencia, ax=None, titulo=None):
    # ax puede ser el eje donde se quiere graficar
    # si no se pasa un ax, entonces se grafica y se muestra
    # tambien se le puede pasar titulo
        
    # Crear un objeto de grafo
    G = nx.from_numpy_matrix(matriz_adyacencia)

    # Obtener el número de nodos
    num_nodos = G.number_of_nodes()

    # Crear un diseño de posición circular
    pos = nx.spring_layout(G)
    
    ver = 0 # no se muestra la figura
    if ax is None:
        ver = 1 # al menos que no se proporcione ax
        # Crear una nueva figura y ejes si no se proporciona ax
        fig, ax = plt.subplots(figsize=(5, 5))

    # Dibujar los nodos
    tamaño_nodos = min(500 /((num_nodos)**1/10), 500) # mientras mas nodos, mas pequeños
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size= tamaño_nodos, ax=ax)
    
    # Dibujar las aristas
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)

    # Dibujar las etiquetas de los nodos
    tamaño_etiqueta = min(10 /((num_nodos)**1/20), 10) # mientras mas nodos, mas pequeños
    nx.draw_networkx_labels(G, pos, font_size= tamaño_etiqueta, font_color='black', ax=ax)
    
    ax.axis('off')
    if titulo:
        ax.set_title(titulo)

    if ver == 1:
        # Mostrar la figura
        plt.show()
        

def dibujar_grafo_from_vector(vector, ax=None, titulo=None):
    
    matriz_adyacencia = vector_to_matriz(vector)
    
    dibujar_grafo_from_matriz(matriz_adyacencia, ax, titulo)
    


def dibujar_grafo_from_lista(lista, ax=None, titulo=None):
    
    matriz_adyacencia = lista_to_matriz(lista)
    
    dibujar_grafo_from_matriz(matriz_adyacencia, ax, titulo)


def dibujar_grafo_from_edge_index(edge_index, ax=None, titulo=None):
    
    matriz_adyacencia = edge_index_to_matriz(edge_index)
    
    dibujar_grafo_from_matriz(matriz_adyacencia, ax, titulo)
    

def energia_from_matriz(matriz_adyacencia):
    # calcular los eigenvalores
    #(usar una funcion que sepa que la matriz es real y simetrica)
    eigenval = np.linalg.eigvalsh(matriz_adyacencia)
    # regresar la energia
    return sum(abs(eigenval))


def energia_from_vector(vector):
    matriz_adyacencia = vector_to_matriz(vector)
    return energia_from_matriz(matriz_adyacencia)

def energia_from_lista(lista):
    matriz_adyacencia = lista_to_matriz(lista)
    return energia_from_matriz(matriz_adyacencia)

def energia_from_edge_index(edge_index):
    matriz_adyacencia = edge_index_to_matriz(edge_index)
    return energia_from_matriz(matriz_adyacencia)

def conectividad_from_matriz(matriz_adyacencia):
    # ver si un grafo es conexo usando la matriz de adyacencia
    num_vertices = matriz_adyacencia.shape[0]

    matriz_potencias = np.array(matriz_adyacencia) # se va ir elevando a potencias
    suma_potencias = np.array(matriz_adyacencia) # se va ir guardando la suma

    # Sumar las potencias de la matriz de adyacencia
    for _ in range(2, num_vertices + 1):
        matriz_potencias = np.dot(matriz_potencias, matriz_adyacencia)
        suma_potencias += matriz_potencias

    # Verificar si la suma de potencias tiene ceros
    if np.any(suma_potencias == 0):
        return 0 # si hay al menos un cero entonces no es conexo
    else: # esta conectado
        return 1

def conectividad_from_vector(vector):
    matriz_adyacencia = vector_to_matriz(vector)
    return conectividad_from_matriz(matriz_adyacencia)

def conectividad_from_lista(lista):
    matriz_adyacencia = lista_to_matriz(lista)
    return conectividad_from_matriz(matriz_adyacencia)

def conectividad_from_edge_index(edge_index):
    matriz_adyacencia = edge_index_to_matriz(edge_index)
    return conectividad_from_matriz(matriz_adyacencia)

def permutar_matriz_aleatorio(matriz):
    # toma una matriz de adyacencia
    # permuta filas y columnas de manera acorde
    # pero la permutacion es aleatorio
    # devuelve la matriz permutada
    # usada para representar mismos grafos con distintas matrices de adyacencia
    
    n = matriz.shape[0]
    # crear matriz de permutacion aleatoria de tamaño n
    P = np.eye(n)         # Crea una matriz identidad de tamaño n
    np.random.shuffle(P)  # Mezcla las filas de la matriz
    
    # Devolver PAP^T
    return np.matmul(np.matmul(P, matriz), P.T)


def grafo_barabasi_m_equal_m0(m = 1, nodos_final = 20, alpha = 1):
    # crea un grafo usando el modelo de barabasi-albert
    # m - parametro del modelo, numero de nodos inicales y de aristas a agregar
    # es decir, se crea por etapas, donde en cada etapa se añade un nodo y se conecta a m preexistentes nodos
    # se considera m = m_0 , es decir, el numero de nodos a los que se conecta cada nodo nuevo es igual 
        # al numero de nodos considerados en un grafo inicial
        # el grafo inicial se considera como el grafo completo de m nodos
    # nodos_final -  es el numero de nodos del grafo resultante
    # alpha - non-linear dependency for the preferential attachment
    
    # crear el grafo inicial (compelto de m nodos)
    grafo_inicial_vector = np.ones((int(m*(m-1)/2)))
    grafo_inicial_lista = vector_to_lista(grafo_inicial_vector)
    
    # hacer iteraciones, en cada una se agrega un nodo al grafo
    iteraciones = nodos_final - m
    
    # si m = 1, entonces el grafo inicial es solo un nodo, sin grado obviamente
    # su representacion en lista es: [[]]
    # esto causa un problema en las iteraciones, crear a mano la primera iteracion
    if m == 1:
        grafo_inicial_lista = [[], [0]] # esta es inevitablemente la primera iteracion
        iteraciones -= 1 # pues se hizo "a mano"
    
     # iniciar la secuencia de iteraciones
    secuencia_grafo_lista = grafo_inicial_lista
    
    for t in range(iteraciones):
        # matriz de adyacencia del grafo en esta iteracion
        matriz_adyacencia = lista_to_matriz(secuencia_grafo_lista)
        # numero de nodos actualmente en el grafo
        n = matriz_adyacencia.shape[0]
        # obtener los grados de cada nodo
        grados = matriz_adyacencia.sum(axis = 1)
        # elevarlo a la alpha
        grados = grados ** alpha
        suma_grados = grados.sum()
        # calcular las probabilidades
        probabilidades = grados / suma_grados
        # se va a crear un nodo y conectarlo a m de los vertices existentes
        # seleccionar de acuerdo con las probabilidades m vertices, de los n existentes
        vertices_a_conectar = np.random.choice(n, p = probabilidades, size = (m), replace = False)
        # hacer que sea una lista ordenada de menor a mayor
        vertices_a_conectar = sorted(list(vertices_a_conectar), reverse = False)
        # actualizar el grafo, con un nuevo vertice y las nuevas conexiones
        secuencia_grafo_lista += [[vertices_a_conectar]]
        
    # se tiene el grafo en forma de lista, hacerlo vector
    grafo_barabasi = lista_to_vector(secuencia_grafo_lista)
    
    return grafo_barabasi