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
    for _ in range(max_iter):
        delta = 0
        new_values = np.copy(values)
        for row in range(env.rows):
            for col in range(env.cols):
                state = env.to_state(row, col)
                if env.is_terminal(state):
                    continue
                action_values = []
                for action in range(env.action_space.n):
                    q = 0
                    for prob, next_state, reward in env.get_transition_prob(state, action):
                        next_row, next_col = env.from_state(next_state)
                        q += prob * (reward + gamma * values[next_row, next_col])
                    action_values.append(q)
                max_value = max(action_values)
                new_values[row, col] = max_value
                delta = max(delta, abs(values[row, col] - new_values[row, col]))
        values = new_values
        if delta < theta:
            break
    # END SOLUTION


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
        new_values = np.copy(values)
        for row in range(env.rows):
            for col in range(env.cols):
                state = env.to_state(row, col)
                if env.is_terminal(state):
                    continue
                action_values = []
                for action in range(env.action_space.n):
                    q = 0
                    for prob, next_state, reward in env.get_transition_prob(state, action):
                        next_row, next_col = env.from_state(next_state)
                        q += prob * (reward + gamma * values[next_row, next_col])
                    action_values.append(q)
                max_value = max(action_values)
                new_values[row, col] = max_value
                delta = max(delta, abs(values[row, col] - new_values[row, col]))
        values = new_values
        if delta < theta:
            break
    # END SOLUTION
    return values