<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FrozenLake Gymnasium Environment</title>
</head>
<body>
    <header>
        <h1>FrozenLake Gymnasium Environment</h1>
    </header>
    <main>
        <p>
          The Frozen Lake problem in the OpenAI Gym environment is a classic reinforcement learning challenge where an agent learns to navigate a grid-like world. The environment is presented as a grid of blocks, where each block can be one of four types:
        </p>
        <ul>
            <li>S (Start): The starting point of the agent.</li>
            <li>F (Frozen): Safe blocks that the agent can step on.</li>
            <li>H (Hole): Dangerous blocks that the agent should avoid, as stepping on them would result in the game ending unsuccessfully.</li>
            <li>G (Goal): The destination block that the agent is trying to reach.</li>
        </ul>
        <p>
          The goal is for the agent to learn a policy to move from the start to the goal, stepping only on the frozen blocks while avoiding the holes. The floor might be slippery, which means the agent might not always move in the intended direction.
        </p>
        <p>
          The maps.py file contains various grid sizes for the environment, ranging from 4x4 to 12x12, with increasing complexity. For example, a 4x4 grid is represented as follows:
        </p>
        <p>
          "FFFG",<br>
          "FHFH",<br>
          "FFFH",<br>
          "SFFH"
        </p>
        <p>
          The Python scripts uses a Q-learning algorithm to train an agent on these maps. It defines a Q-table to keep track of the value of taking a certain action in a particular state. another Approuch script "Assignmnet2_38336529_FrozenLake_Hyper" includes hyperparameters such as learning rate, discount factor, and epsilon decay rate to adjust the balance between exploration (trying out new actions) and exploitation (using known information to maximize the reward). Over time, the agent explores less and relies more on the learned values in the Q-table.
        </p>
        <p>
          The agent's interactions with the environment, along with its rewards, are graphed to illustrate the learning progress over time.
        </p>
    </main>
    <footer>
        <p>&copy; 2024 Dewald Oosthuizen 38336529</p>
    </footer>
</body>
</html>
