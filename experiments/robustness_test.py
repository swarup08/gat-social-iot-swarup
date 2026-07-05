"""
====================================================
Robustness Test
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================

Test robustness of the Dynamic RL pruning policy
under different attack scenarios.
"""

from experiments.evaluate_dynamic_rl import evaluate_dynamic_rl


def test_initial_infections():

    print("=" * 60)
    print("Robustness Test : Initial Infected Nodes")
    print("=" * 60)

    seeds = [3, 5, 7]

    for seed_count in seeds:

        print(f"\nInitial Infected Nodes : {seed_count}")

        result = evaluate_dynamic_rl(
            verbose=False,
            num_seeds=seed_count
        )

        print(
            f"Infection={result['infection']:.3f} | "
            f"Connectivity={result['connectivity']:.3f} | "
            f"Reward={result['total_reward']:.3f}"
        )
def test_infection_probability():

    print("\n" + "=" * 60)
    print("Robustness Test : Infection Probability")
    print("=" * 60)

    probabilities = [0.15, 0.25, 0.35]

    for prob in probabilities:

        print(f"\nInfection Probability : {prob}")

        result = evaluate_dynamic_rl(
            verbose=False,
            infection_prob=prob
        )

        print(
            f"Infection={result['infection']:.3f} | "
            f"Connectivity={result['connectivity']:.3f} | "
            f"Reward={result['total_reward']:.3f}"
        )

def main():

    test_initial_infections()

    test_infection_probability()

if __name__ == "__main__":

    main()