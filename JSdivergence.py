import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt

def compute_histogram(series, bins=30):
    """
    Calcola l'istogramma normalizzato di una serie temporale.

    Args:
        series (array-like): Serie temporale.
        bins (int): Numero di bin per l'istogramma.

    Returns:
        np.ndarray: Istogramma normalizzato.
    """
    hist, _ = np.histogram(series, bins=bins, density=True)
    return hist

def jensen_shannon_divergence(p, q):
    """
    Calcola la Jensen-Shannon Divergence tra due distribuzioni p e q.

    Args:
        p (array-like): Prima distribuzione di probabilità.
        q (array-like): Seconda distribuzione di probabilità.

    Returns:
        float: Jensen-Shannon Divergence tra p e q.
    """
    P = compute_histogram(p, bins=30)
    Q = compute_histogram(q, bins=30)

    # Convertire in array numpy e normalizzare
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Normalizzazione delle distribuzioni
    P /= P.sum()
    Q /= Q.sum()

    # Calcolare la media delle distribuzioni
    M = 0.5 * (P + Q)

    # Calcolare le KLD
    kl_pm = kl_div(P, M)  # KLD tra P e M
    kl_qm = kl_div(Q, M)  # KLD tra Q e M

    # JSD è la media delle KLD
    jsd = 0.5 * (kl_pm.sum() + kl_qm.sum())
    
    return jsd

"""
# Esempio di utilizzo con due serie temporali esistenti
# Sostituisci queste serie con i tuoi dati esistenti
# Esempio di due serie temporali sinusoidali
t = np.linspace(0, 2 * np.pi, 1000)  # Tempo
series1 = np.sin(t)                   # Serie 1: Sinusoidale
series2 = 0.5 * np.sin(t + np.pi / 4)  # Serie 2: Sinusoidale con fase diversa

# Calcolare la JSD tra i due istogrammi
jsd_value = jensen_shannon_divergence(series1, series2)
print(f"Jensen-Shannon Divergence tra le due serie temporali: {jsd_value}")

# Visualizzazione delle serie e dei loro istogrammi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(series1, bins=30, alpha=0.5, label='Serie 1', color='blue', density=True)
plt.hist(series2, bins=30, alpha=0.5, label='Serie 2', color='orange', density=True)
plt.title('Istogrammi delle Serie Temporali')
plt.xlabel('Valore')
plt.ylabel('Densità')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, series1, label='Serie 1', alpha=0.7)
plt.plot(t, series2, label='Serie 2', alpha=0.7)
plt.title('Serie Temporali')
plt.xlabel('Tempo')
plt.ylabel('Valore')
plt.legend()

plt.tight_layout()
plt.show()
"""