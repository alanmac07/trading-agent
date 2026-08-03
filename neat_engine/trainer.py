"""
neat_engine/trainer.py

Run: python -m neat_engine.trainer --generations 200
Resume: python -m neat_engine.trainer --generations 200 --resume checkpoints/neat-checkpoint-50

Requires data/train_extended.csv to exist first:
    python -m neat_engine.build_dataset
"""

from __future__ import annotations

import argparse
import os
import pickle
import random

import neat

from .episodes import build_episodes, sample_episodes_for_generation
from .fitness import fitness_for_genome

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
TRAIN_CSV = "data/train_extended.csv"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

EPISODES_PER_GENERATION = 6  # only used when NOT running --parallel, see episodes.py docstring


class BestGenomeSaver(neat.reporting.BaseReporter):
    """
    Writes the best genome seen so far to results/best_genome.pkl every time
    it improves, AND every `every_n` generations regardless -- so you can
    Ctrl+C mid-run and still have something to load into evaluate_holdout.py,
    instead of only getting a saved genome once the whole run finishes.
    """

    def __init__(self, every_n: int = 10):
        self.best = None
        self.every_n = every_n
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        improved = self.best is None or best_genome.fitness > self.best.fitness
        if improved:
            self.best = best_genome
        if improved or (self.generation % self.every_n == 0):
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(os.path.join(RESULTS_DIR, "best_genome.pkl"), "wb") as f:
                pickle.dump(self.best, f)


def make_eval_function(all_episodes, episodes_per_gen: int):
    rng = random.Random(42)

    def eval_genomes(genomes, config):
        episodes = sample_episodes_for_generation(all_episodes, episodes_per_gen, rng)
        for _, genome in genomes:
            genome.fitness = fitness_for_genome(genome, config, episodes)

    return eval_genomes


# --- parallel path -----------------------------------------------------
# Module-level (not a closure) so it's picklable/importable fresh in each
# worker process -- required for multiprocessing on Windows (spawn), which
# does NOT inherit globals set in the parent after the pool is created.
_episodes_cache = None


def _get_all_episodes():
    global _episodes_cache
    if _episodes_cache is None:
        _episodes_cache = build_episodes(TRAIN_CSV)
    return _episodes_cache


def eval_single_genome(genome, config):
    episodes = _get_all_episodes()  # full set -- no sampling once parallel
    return fitness_for_genome(genome, config, episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--parallel", action="store_true", help="use all CPU cores, full episode set")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH,
    )

    if args.resume:
        pop = neat.Checkpointer.restore_checkpoint(args.resume)
    else:
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=10,
        filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-"),
    ))
    genome_saver = BestGenomeSaver(every_n=10)
    pop.add_reporter(genome_saver)

    if args.parallel:
        all_episodes = _get_all_episodes()
        print(f"Loaded {len(all_episodes)} total training episodes across all tickers.")
        print(f"Parallel mode: {args.workers} workers, evaluating ALL episodes every generation.\n")
        pe = neat.ParallelEvaluator(args.workers, eval_single_genome)
        winner = pop.run(pe.evaluate, n=args.generations)
    else:
        all_episodes = build_episodes(TRAIN_CSV)
        print(f"Loaded {len(all_episodes)} total training episodes across all tickers.")
        print(f"Sampling {EPISODES_PER_GENERATION} episodes/generation for fitness evaluation.\n")
        eval_genomes = make_eval_function(all_episodes, EPISODES_PER_GENERATION)
        winner = pop.run(eval_genomes, n=args.generations)

    with open(os.path.join(RESULTS_DIR, "best_genome.pkl"), "wb") as f:
        pickle.dump(winner, f)

    with open(os.path.join(RESULTS_DIR, "fitness_history.csv"), "w") as f:
        f.write("generation,best_fitness,avg_fitness\n")
        for gen, (best, avg) in enumerate(zip(stats.get_fitness_stat(max), stats.get_fitness_mean())):
            f.write(f"{gen},{best},{avg}\n")

    print(f"\nDone. Best genome saved to {RESULTS_DIR}/best_genome.pkl")
    print(f"Best fitness achieved: {winner.fitness:.3f}")


if __name__ == "__main__":
    main()
