# RL and Interpretability (CE888 Assignment 2)

Modern Reinforcement Learning helps agents learn how to act using complex patterns of text, sound and video and it's slowly moving away from research and making inroads to traditional industries (e.g., creating game NPC characters). The high dimensionality of the input space makes it very hard to interpret why an agent preferred one action over another. In this project we will try to transfer some novel methods from supervised learning to Reinforcement Learning in order to interpret why agents make certain decisions. We will use already existing Atari game playing agents and try to interpret their actions profile in real time, effectively "seeing" through the agent's eyes.

1. See https://github.com/ozzstrich/CE888_Assignment

2. Pass the observations to LIME and get an interpreted image/observation. Save the interpreted images.

3. Calculate the optical flow between the first image in an observation and the last image.

4. Use one of the unsupervised learning algorithms from sci-kit learn to break down your data in various segments - are there clear clusters being formed? What is in each cluster?

5. Create a video of interpreted agent actions and upload on youtube (optional)
