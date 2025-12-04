"""
Score Logger for Reinforcement Learning
---------------------------------------
This utility logs episode scores, computes rolling averages,
and plots training progress over time.
"""

import os
import csv
from statistics import mean
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
SCORES_DIR = "./scores"
SCORES_CSV_PATH = os.path.join(SCORES_DIR, "scores.csv")
SCORES_PNG_PATH = os.path.join(SCORES_DIR, "scores.png")
SOLVED_CSV_PATH = os.path.join(SCORES_DIR, "solved.csv")
SOLVED_PNG_PATH = os.path.join(SCORES_DIR, "solved.png")

AVERAGE_SCORE_TO_SOLVE = 475
CONSECUTIVE_RUNS_TO_SOLVE = 100


class ScoreLogger:
    """Logs episode scores and generates plots."""

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)

        os.makedirs(SCORES_DIR, exist_ok=True)

        # Clean old score files
        for path in [SCORES_PNG_PATH, SCORES_CSV_PATH]:
            if os.path.exists(path):
                os.remove(path)

    def add_score(self, score: float, run: int):
        """Add a score for a given episode and update logs/plots."""
        self._save_csv(SCORES_CSV_PATH, score)
        self._save_png(
            input_path=SCORES_CSV_PATH,
            output_path=SCORES_PNG_PATH,
            x_label="Episodes",
            y_label="Scores",
            average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
            show_goal=True,
            show_trend=True, 
            show_legend=True,
        )

        self.scores.append(score)
        mean_score = mean(self.scores)
        print(f"Scores: (min: {min(self.scores)}, avg: {mean_score:.2f}, max: {max(self.scores)})")

        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solved_in = run - CONSECUTIVE_RUNS_TO_SOLVE
            print(f"Solved in {solved_in} runs, {run} total runs.")
            self._save_csv(SOLVED_CSV_PATH, solved_in)
            self._save_png(
                input_path=SOLVED_CSV_PATH,
                output_path=SOLVED_PNG_PATH,
                x_label="Trials",
                y_label="Steps before solved",
                average_of_n_last=None,
                show_goal=False,
                show_trend=False,
                show_legend=False,
            )
            # exit(0) 

    def _save_png(self, input_path, output_path,
                  x_label, y_label,
                  average_of_n_last,
                  show_goal, show_trend, show_legend):
        """Generate a PNG chart for score evolution."""
        if not os.path.exists(input_path):
            return

        x, y = [], []
        with open(input_path, "r") as scores_file:
            reader = csv.reader(scores_file)
            data = list(reader)
            for i, row in enumerate(data):
                x.append(i)
                y.append(float(row[0]))

        plt.subplots()
        plt.plot(x, y, label="Score per Episode")

        if average_of_n_last is not None and len(x) > 0:
            avg_range = min(average_of_n_last, len(x))
            plt.plot(
                x[-avg_range:],
                [np.mean(y[-avg_range:])] * avg_range,
                linestyle="--",
                label=f"Average of last {avg_range}",
            )

        if show_goal:
            # plot the target score line
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=f"Goal ({AVERAGE_SCORE_TO_SOLVE} Avg)")

        if show_trend and len(x) > 1:
            y_trend = []
            current_block_scores = [] 
            
            y_np = np.array(y)
            
            for i in range(len(y_np)):
                current_block_scores.append(y_np[i])
                
                current_avg = np.mean(current_block_scores)
                y_trend.append(current_avg)
                
                if (i + 1) % 100 == 0:
                    current_block_scores = []
            
            plt.plot(x, y_trend, linestyle="-.", label="Trend (100-ep Reset Avg)")

        plt.title(f"{self.env_name} - Training Progress")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if show_legend:
            plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score: float):
        """Append a score to the given CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([score])