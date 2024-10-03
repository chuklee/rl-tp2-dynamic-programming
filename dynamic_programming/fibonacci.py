"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    # BEGIN SOLUTION
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    # END SOLUTION


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.


def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """

    # BEGIN SOLUTION
    memo = {}  # Dictionnaire pour mémoriser les résultats intermédiaires
    
    def fib_memo(n: int) -> int:
        # Si le résultat est déjà mémorisé, le retourner directement
        if n in memo:
            return memo[n]
        
        # Cas de base : F(0) = 0 et F(1) = 1
        if n == 0:
            return 0
        elif n == 1:
            return 1
        
        # Sinon, calculer de manière récursive et mémoriser le résultat
        memo[n] = fib_memo(n - 1) + fib_memo(n - 2)
        return memo[n]
    
    return fib_memo(n)
    # END SOLUTION
