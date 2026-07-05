from src.rl.environment import GraphPruningEnv


def main():

    env = GraphPruningEnv()

    state = env.reset()

    print("=" * 50)
    print("Initial State")
    print(state)
    print("=" * 50)

    next_state, reward, done, info = env.step(0)

    print("\nAfter First Action")

    print("Next State :", next_state)
    print("Reward     :", reward)
    print("Done       :", done)
    print("Info       :", info)


if __name__ == "__main__":
    main()