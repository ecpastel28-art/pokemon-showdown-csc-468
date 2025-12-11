# learning_agent.py

import asyncio
import random
from collections import defaultdict
from math import sqrt

from poke_env import RandomPlayer
from poke_env.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


# ---------------------------------------------------------------------------
# Q-Learning Player
# ---------------------------------------------------------------------------

class QLearningPlayer(Player):
    """
    Simple but upgraded Q-learning agent for Pokémon Showdown via poke-env.

    STATE:
        (my_hp_bucket, opp_hp_bucket, my_remaining, opp_remaining)

    ACTIONS:
        Index into battle.available_moves (0..len-1)

    REWARD:
        Δ[(# opponent fainted) - (# my fainted)]  between steps
        (i.e. reward is the CHANGE in KO difference, not the absolute value)

    EXPLORATION:
        Epsilon decays from epsilon_start -> epsilon_end over a number
        of steps (epsilon_decay_steps).

    HEURISTICS:
        When scoring actions, we use:
            Q(s, a) + small_heuristic_bonus(move, game_state)

        Heuristics are intentionally shallow to avoid breaking things:
        - Slight bonus for higher base power.
        - Slight penalty for poor accuracy when we're ahead.
        - Slight bonus for priority moves when both sides are low.
    """

    def __init__(
        self,
        epsilon_start: float = 0.6,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        alpha: float = 0.3,
        gamma: float = 0.95,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self._steps = 0

        self.alpha = alpha    # learning rate
        self.gamma = gamma    # discount factor

        # Q-table: Q[state][action_index] -> value
        self.q = defaultdict(lambda: defaultdict(float))

        # Last transition info
        self._last_state = None     # tuple
        self._last_action = None    # int (index into available_moves)
        self._last_ko_diff = 0      # previous (opp_fainted - my_fainted)

    # ---------------- State & reward helpers ----------------

    def _hp_bucket(self, pokemon):
        """Discretize HP into 3 buckets: 2 = healthy, 1 = mid, 0 = low/KO."""
        if pokemon is None:
            return 0
        frac = pokemon.current_hp_fraction  # 0..1
        if frac > 2 / 3:
            return 2
        elif frac > 1 / 3:
            return 1
        else:
            return 0

    def embed_battle(self, battle):
        """Map the battle to a compact discrete state."""
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon

        my_hp = self._hp_bucket(me)
        opp_hp = self._hp_bucket(opp)

        my_remaining = sum(not p.fainted for p in battle.team.values())
        opp_remaining = sum(not p.fainted for p in battle.opponent_team.values())

        return (my_hp, opp_hp, my_remaining, opp_remaining)

    def _ko_diff(self, battle):
        """Return (# opp fainted) - (# my fainted)."""
        my_fainted = sum(p.fainted for p in battle.team.values())
        opp_fainted = sum(p.fainted for p in battle.opponent_team.values())
        return opp_fainted - my_fainted

    def _reward(self, battle):
        """
        Reward = change in KO difference since last step:
            r_t = [(opp_fainted - my_fainted)_t] - [(opp_fainted - my_fainted)_{t-1}]
        """
        current = self._ko_diff(battle)
        reward = current - self._last_ko_diff
        self._last_ko_diff = current
        return reward

    # ---------------- Epsilon schedule ----------------

    def _update_epsilon(self):
        """Linearly decay epsilon from epsilon_start to epsilon_end."""
        self._steps += 1
        frac = min(1.0, self._steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ---------------- Heuristic bonus for moves ----------------

    def _move_bonus(self, move, battle):
        """
        Small heuristic bonus for a given move.
        We keep it VERY simple and safe (no type calculations).
        """
        bonus = 0.0

        # Extract some lightweight game context
        my_remaining = sum(not p.fainted for p in battle.team.values())
        opp_remaining = sum(not p.fainted for p in battle.opponent_team.values())
        ahead_in_kos = self._ko_diff(battle) > 0

        # 1) Base power: stronger hits are generally good
        if move.base_power is not None:
            if move.base_power >= 100:
                bonus += 0.12
            elif move.base_power >= 70:
                bonus += 0.06

        # 2) Accuracy & consistency: penalize lower accuracy when we're ahead
        if move.accuracy is not None and ahead_in_kos:
            if move.accuracy < 0.85:
                bonus -= 0.12
            elif move.accuracy < 0.9:
                bonus -= 0.08

        # 3) Priority: useful when both sides are low on resources
        if move.priority > 0 and my_remaining <= 2 and opp_remaining <= 2:
            bonus += 0.10

        # 4) PP management: avoid spamming the last few PP early game
        if move.max_pp and move.current_pp is not None:
            if move.current_pp <= 2 and opp_remaining >= 3:
                bonus -= 0.05

        return bonus

    # ---------------- Core Q-learning + policy ----------------

    def choose_move(self, battle):
        """
        Called by poke-env every turn.

        1. Build current state & reward.
        2. TD-update using previous (state, action).
        3. Update epsilon (decay).
        4. ε-greedy over available_moves, using Q + heuristic bonus.
        5. Return a BattleOrder via self.create_order(...).
        """
        # Embed current state
        state = self.embed_battle(battle)
        reward = self._reward(battle)

        # ----- TD update from last step -----
        if self._last_state is not None and self._last_action is not None:
            next_q_values = self.q[state]
            best_next_q = max(next_q_values.values()) if next_q_values else 0.0

            old_q = self.q[self._last_state][self._last_action]
            new_q = old_q + self.alpha * (
                reward + self.gamma * best_next_q - old_q
            )
            self.q[self._last_state][self._last_action] = new_q

        # ----- Forced switch: no moves available -----
        if not battle.available_moves:
            if battle.available_switches:
                target = random.choice(battle.available_switches)
                return self.create_order(target)
            else:
                # Fallback to parent implementation (extremely rare case)
                return super().choose_move(battle)

        # ----- Update epsilon (exploration rate) -----
        self._update_epsilon()

        # ----- Q-learning: choose among attack moves -----
        n_actions = len(battle.available_moves)
        actions = list(range(n_actions))

        # ε-greedy selection with heuristic bonus
        if random.random() < self.epsilon:
            action_idx = random.choice(actions)
        else:
            q_state = self.q[state]
            # Score = Q + move heuristic
            best_score = None
            best_action = None
            for a in actions:
                move = battle.available_moves[a]
                q_val = q_state[a]
                bonus = self._move_bonus(move, battle)
                score = q_val + bonus
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = a
            action_idx = best_action if best_action is not None else random.choice(actions)

        self._last_state = state
        self._last_action = action_idx

        chosen_move = battle.available_moves[action_idx]
        return self.create_order(chosen_move)


# ---------------------------------------------------------------------------
# Training + evaluation loop
# ---------------------------------------------------------------------------

def _print_summary(title, n_battles, wins, losses, ties):
    win_rate = wins / n_battles if n_battles else 0.0
    loss_rate = losses / n_battles if n_battles else 0.0
    tie_rate = ties / n_battles if n_battles else 0.0

    print()
    print(f"=== {title} ===")
    print(f"  Battles: {n_battles}")
    print(f"  Wins:    {wins:3d} ({win_rate:.1%})")
    print(f"  Losses:  {losses:3d} ({loss_rate:.1%})")
    print(f"  Ties:    {ties:3d} ({tie_rate:.1%})")

    # Win-rate bar
    bar_len = 30
    filled = int(bar_len * win_rate)
    bar = "[" + "#" * filled + "-" * (bar_len - filled) + f"] {win_rate:.1%}"
    print(f"\n  Win-rate bar:")
    print(f"  {bar}")

    # Approximate 95% CI for win prob (treating ties as non-wins)
    if n_battles > 0:
        p_hat = wins / n_battles
        se = sqrt(p_hat * (1 - p_hat) / n_battles)
        lower = max(0.0, p_hat - 1.96 * se)
        upper = min(1.0, p_hat + 1.96 * se)
        print(f"\n  Approx. 95% CI for win rate: [{lower:.1%}, {upper:.1%}]")
    print()


async def train_and_evaluate():
    # Use the same server_configuration style as example_script
    server_cfg = LocalhostServerConfiguration

    learner = QLearningPlayer(
        battle_format="gen9randombattle",
        server_configuration=server_cfg,
        start_timer_on_battle_start=False,
        # exploration schedule
        epsilon_start=0.6,
        epsilon_end=0.05,
        epsilon_decay_steps=10_000,
        # learning hyperparams
        alpha=0.3,
        gamma=0.95,
    )
    random_opp = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=server_cfg,
        start_timer_on_battle_start=False,
    )

    # -------- TRAINING --------
    N_TRAIN = 1500  # increase for stronger agent
    print("=" * 60)
    print(f"Training Q-learning agent for {N_TRAIN} battles vs Random...")
    print("=" * 60)

    await learner.battle_against(random_opp, n_battles=N_TRAIN)

    train_wins = learner.n_won_battles
    train_losses = random_opp.n_won_battles
    train_ties = max(N_TRAIN - (train_wins + train_losses), 0)

    _print_summary("TRAINING SUMMARY", N_TRAIN, train_wins, train_losses, train_ties)

    # -------- EVALUATION --------
    learner.epsilon = 0.0  # greedy policy, no exploration
    learner.reset_battles()   # keep Q-table, reset counters
    random_opp.reset_battles()

    N_EVAL = 300
    print("=" * 60)
    print(f"Evaluating trained learner for {N_EVAL} battles vs Random...")
    print("=" * 60)

    await learner.battle_against(random_opp, n_battles=N_EVAL)

    eval_wins = learner.n_won_battles
    eval_losses = random_opp.n_won_battles
    eval_ties = max(N_EVAL - (eval_wins + eval_losses), 0)

    _print_summary("EVALUATION SUMMARY", N_EVAL, eval_wins, eval_losses, eval_ties)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(train_and_evaluate())

