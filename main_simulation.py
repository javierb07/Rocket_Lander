"""
Author: Reuben Ferrante
Date:   16/08/2017
Description: This is the  main running point of the simulation. Set settings, algorithm, episodes,...
Modified by: Javier Belmonte
Date: 04/26/2020
Added: Fuzzy PID control, comparison metrics (time, fuel, goal), data saving for analysis
"""

from environments.rocketlander import RocketLander
from constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT
import numpy as np
import time
import csv

if __name__ == "__main__":
    # Settings holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': '(6000, -10000)'}  # (6000, -10000)}random

    env = RocketLander(settings)
    s = env._reset()
    random_environment = False   # Set to true to simulate movement of barge and wind
    verbose = True  # Set to true to print performance metrics to console

    from control_and_ai.pid import PID_Benchmark, Fuzzy_PID

    # Initialize the PID algorithm
    pid = PID_Benchmark()

    # Initialize the Fuzzy PID algorithm
    fuzz_pid = Fuzzy_PID()
    show_mf = False     # Decide to show membership functions of fuzzy pid or not
    if show_mf:
        fuzz_pid.Fe_PID.show_mf_groups()
        fuzz_pid.Fs_theta_PID.show_mf_groups()
        fuzz_pid.psi_PID.show_mf_groups()

    metrics = {"pid": {"fuel": 0, "x_final": 0, "time": 0}, "fuzzy": {"fuel": 0, "x_final": 0, "time": 0}}
    saved_pid_metrics = []
    saved_fuzzy_metrics = []

    algorithms = [pid, fuzz_pid]    # List of algorithms to simulate and control the rocket

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.05
    total_reward = 0
    episode_number = 10 # Number of episodes to simulate

    for episode in range(episode_number):   # Simulate each episode
        for algorithm in algorithms:
            start_time = time.time()
            while True:
                a = algorithm.pid_algorithm(s) # pass the state to the algorithm, get the actions
                # Step through the simulation (1 step). Refer to Simulation Update in constants.py
                s, r, done, info = env._step(a)
                total_reward += r   # Accumulate reward
                # -------------------------------------
                # Optional render
                env._render()
                # Draw the target
                env.draw_marker(env.landing_coordinates[0], env.landing_coordinates[1])
                # Refresh render
                env.refresh(render=False)

                if random_environment:
                    # When should the barge move? Water movement, dynamics etc can be simulated here.
                    if s[LEFT_GROUND_CONTACT] == 0 and s[RIGHT_GROUND_CONTACT] == 0:
                        env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                        # Random Force on rocket to simulate wind.
                        env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                        env.apply_random_y_disturbance(epsilon=0.005)

                # Touch down or pass abs(THETA_LIMIT)
                if done:
                    if algorithm == pid:
                        metrics["pid"]["time"] = time.time() - start_time
                        metrics["pid"]["fuel"] = env.get_consumed_fuel()
                        metrics["pid"]["x_final"] = abs(s[0])
                        temp_dic = metrics["pid"].copy()
                        saved_pid_metrics.append(temp_dic)
                    elif algorithm == fuzz_pid:
                        metrics["fuzzy"]["time"] = time.time() - start_time
                        metrics["fuzzy"]["fuel"] = env.get_consumed_fuel()
                        metrics["fuzzy"]["x_final"] = abs(s[0])
                        temp_dic = metrics["fuzzy"].copy()
                        saved_fuzzy_metrics.append(temp_dic)

                    total_reward = 0
                    env._reset()
                    break

    if verbose:
        for episode in range(episode_number):
            print("==================================")
            print('Episode:\t{}'.format(episode+1))
            print("Type\tTime\tx_final\tfuel")
            pid_e = saved_pid_metrics[episode]
            fuzzy_e = saved_fuzzy_metrics[episode]
            print("PID:\t{:.2f}\t{:.2f}\t{:.2f}".format(pid_e["time"],pid_e["x_final"], pid_e["fuel"]))
            print("Fuzzy:\t{:.2f}\t{:.2f}\t{:.2f}".format(fuzzy_e["time"],fuzzy_e["x_final"], fuzzy_e["fuel"]))
            print("==================================")

    # Write results to file for analysis
    w = csv.writer(open("results.csv", "w"))
    w.writerow(["PID"])
    try:
        for idx, metric in enumerate(saved_pid_metrics):
            w.writerow(["Episode", idx+1])
            for key, val in metric.items():
                w.writerow([key, val])
        w.writerow(["Fuzzy PID"])
        for idx, metric in enumerate(saved_fuzzy_metrics):
            w.writerow(["Episode", idx+1])
            for key, val in metric.items():
                w.writerow([key, val])
    except NameError:
        pass
