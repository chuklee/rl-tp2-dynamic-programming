import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    for iteration in range(max_iter):
        # Copie de la fonction de valeur actuelle pour la comparer après la mise à jour
        V_prev = np.copy(values)
        
        # Pour chaque état
        for state in range(mdp.observation_space.n):
            # On calcule la valeur de chaque action
            action_values = []
            for action in range(mdp.action_space.n):
                next_state, reward, done = mdp.P[state][action]
                action_value = reward + gamma * V_prev[next_state]  # Formule de Bellman
                action_values.append(action_value)
            
            # Mise à jour de la fonction de valeur de l'état en choisissant l'action qui maximise la valeur
            values[state] = max(action_values)
        
        # Vérification de la convergence : si la différence est suffisamment petite, on arrête
        if np.allclose(values, V_prev, atol=1e-6):
            break
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for iteration in range(max_iter):
        delta = 0
        for row in range(4):
            for col in range(4):
                if env.grid[row, col] in {"P", "N", "W"}:
                    continue  # Ici je skip parce que c'est terminal ou mur
                old_value = values[row, col]
                action_values = []
                for action in range(4):  # 0: Up, 1: Down, 2: Left, 3: Right
                    next_row, next_col = row, col
                    if action == 0:  # Up
                        next_row = max(0, row - 1)
                    elif action == 1:  # Down
                        next_row = min(3, row + 1)
                    elif action == 2:  # Left
                        next_col = max(0, col - 1)
                    elif action == 3:  # Right
                        next_col = min(3, col + 1)
                    
                    if env.grid[next_row, next_col] == "W":
                        next_row, next_col = row, col
                    
                    reward = 1 if env.grid[next_row, next_col] == "P" else -1 if env.grid[next_row, next_col] == "N" else 0
                    action_values.append(reward + gamma * values[next_row, next_col])
                values[row, col] = max(action_values)
                delta = max(delta, abs(old_value - values[row, col]))
        
        if delta < theta:
            break
    # END SOLUTION
    return values


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for _ in range(max_iter):
        delta = 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if env.grid[row, col] in {"P", "N"}:  # Check for terminal states
                    continue
                env.current_position = (row, col)  # (Faut mettre à jour la position actuelle)
                old_value = values[row, col]
                action_values = []
                for action in range(env.action_space.n):
                    q = 0
                    next_states = env.get_next_states(action)
                    for next_state, reward, prob, _, _ in next_states:
                        next_row, next_col = next_state  # Unpack the tuple directly
                        q += prob * env.moving_prob[row, col, action] * (reward + gamma * values[next_row, next_col])
                    action_values.append(q)
                values[row, col] = max(action_values)
                delta = max(delta, abs(old_value - values[row, col]))
        if delta < theta:
            break
    # END SOLUTION
    return values