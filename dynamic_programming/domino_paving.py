# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    if n == 0:
        return 1
    if n == 1:
        return 0
    if n == 2:
        return 3

    f = [0] * (n + 1)
    g = [0] * (n + 1)

    f[0] = 1
    f[1] = 0
    f[2] = 3
    g[0] = 0
    g[1] = 1
    g[2] = 0

    for i in range(3, n + 1):
        f[i] = f[i - 2] + 2 * g[i - 1]
        g[i] = f[i - 1] + g[i - 2]

    return f[n]
    # END SOLUTION
