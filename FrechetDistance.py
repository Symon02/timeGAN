import numpy as np

def euclidean_distance(p1, p2):
    """Calcola la distanza euclidea tra due punti p1 e p2."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def frechet_distance(P, Q):
    """
    Calcola la Frechet Distance tra due sequenze P e Q.
    P e Q sono array bidimensionali con le coordinate dei punti della traiettoria.
    """
    n = len(P)
    m = len(Q)
    ca = np.ones((n, m)) * -1  # Array per memorizzare i risultati intermedi

    def recursive_calculate(i, j):
        """Funzione ricorsiva per calcolare la distanza di Frechet con memoizzazione."""
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = euclidean_distance(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(recursive_calculate(i-1, 0), euclidean_distance(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(recursive_calculate(0, j-1), euclidean_distance(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(
                    recursive_calculate(i-1, j),
                    recursive_calculate(i-1, j-1),
                    recursive_calculate(i, j-1)
                ),
                euclidean_distance(P[i], Q[j])
            )
        else:
            ca[i, j] = float('inf')

        return ca[i, j]

    return recursive_calculate(n-1, m-1)

# Esempio di utilizzo
P = [(0, 0), (1, 1), (2, 2), (3, 3)]
Q = [(0, 0), (1, 2), (2, 2), (4, 4)]

distance = frechet_distance(P, Q)
print(f"La Frechet Distance tra le sequenze è: {distance}")
