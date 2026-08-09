"""
neat_engine/trainer.py

Run:
  python -m neat_engine.trainer --market stocks --generations 200 --parallel
  python -m neat_engine.trainer --market crypto --generations 200 --parallel
  python -m neat_engine.trainer --market macro --generations 200 --parallel

Universal:
  python -m neat_engine.trainer --generations 200 --parallel

Resume:
  python -m neat_engine.trainer --market stocks --generations 200 --resume checkpoints/neat-checkpoint-stocks-50
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
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

TRAIN_CSV = os.environ.get("NEAT_TRAIN_CSV", "data/train_extended.csv")
BEST_GENOME_PATH = os.environ.get("NEAT_BEST_GENOME_PATH", os.path.join(RESULTS_DIR, "best_genome.pkl"))

EPISODES_PER_GENERATION = 6  # only used when NOT running --parallel, see episodes.py docstring


class BestGenomeSaver(neat.reporting.BaseReporter):
    """
    Writes the best genome seen so far to the target pickle path every time
    it improves, AND every `every_n` generations regardless -- so you can
    Ctrl+C mid-run and still have something to load into evaluate_holdout.py.
    """

    def __init__(self, save_path: str = BEST_GENOME_PATH, every_n: int = 10):
        self.best = None
        self.save_path = save_path
        self.every_n = every_n
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        improved = self.best is None or best_genome.fitness > self.best.fitness
        if improved:
            self.best = best_genome
        if improved or (self.generation % self.every_n == 0):
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with open(self.save_path, "wb") as f:
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
    train_csv = os.environ.get("NEAT_TRAIN_CSV", TRAIN_CSV)
    if _episodes_cache is None:
        _episodes_cache = build_episodes(train_csv)
    return _episodes_cache


def eval_single_genome(genome, config):
    episodes = _get_all_episodes()  # full set -- no sampling once parallel
    return fitness_for_genome(genome, config, episodes)


def main():
    parser = argparse.ArgumentParser(description="Train NEAT trading agents on market regimes.")
    parser.add_argument("--generations", type=int, default=150, help="Number of generations to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--parallel", action="store_true", help="Use all CPU cores, full episode set")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Worker count for parallel evaluation")
    parser.add_argument("--market", type=str, default=None, choices=["stocks", "crypto", "macro", None], help="Market regime")
    args = parser.parse_args()

    market_str = args.market.lower() if args.market else None
    if market_str:
        train_csv = f"data/train_{market_str}.csv"
        best_genome_path = os.path.join(RESULTS_DIR, f"best_genome_{market_str}.pkl")
        checkpoint_prefix = os.path.join(CHECKPOINT_DIR, f"neat-checkpoint-{market_str}-")
        fitness_history_path = os.path.join(RESULTS_DIR, f"fitness_history_{market_str}.csv")
    else:
        train_csv = "data/train_extended.csv"
        best_genome_path = os.path.join(RESULTS_DIR, "best_genome.pkl")
        checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
        fitness_history_path = os.path.join(RESULTS_DIR, "fitness_history.csv")

    # Set environment variables for spawned subprocesses in parallel mode
    os.environ["NEAT_TRAIN_CSV"] = train_csv
    os.environ["NEAT_BEST_GENOME_PATH"] = best_genome_path

    global TRAIN_CSV, BEST_GENOME_PATH
    TRAIN_CSV = train_csv
    BEST_GENOME_PATH = best_genome_path

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training dataset '{train_csv}' not found. Run 'python -m neat_engine.build_dataset' first.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== NEAT Training Configuration ===")
    print(f"Target Market:       {market_str.upper() if market_str else 'UNIVERSAL'}")
    print(f"Training Data:       {train_csv}")
    print(f"Target Genome Path:  {best_genome_path}")
    print(f"Checkpoint Prefix:   {checkpoint_prefix}")
    print(f"Generations:         {args.generations}")
    print(f"Parallel Evaluator:  {args.parallel} ({args.workers} workers)")
    print(f"===================================\n")

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        CONFIG_PATH,
    )

    if args.resume:
        print(f"Restoring population from checkpoint: {args.resume}")
        pop = neat.Checkpointer.restore_checkpoint(args.resume)
    else:
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=10,
        filename_prefix=checkpoint_prefix,
    ))
    genome_saver = BestGenomeSaver(save_path=best_genome_path, every_n=10)
    pop.add_reporter(genome_saver)

    if args.parallel:
        all_episodes = _get_all_episodes()
        print(f"Loaded {len(all_episodes)} total training episodes from {train_csv}.")
        print(f"Parallel mode: {args.workers} workers, evaluating ALL episodes every generation.\n")
        pe = neat.ParallelEvaluator(args.workers, eval_single_genome)
        winner = pop.run(pe.evaluate, n=args.generations)
    else:
        all_episodes = build_episodes(train_csv)
        print(f"Loaded {len(all_episodes)} total training episodes from {train_csv}.")
        print(f"Sampling {EPISODES_PER_GENERATION} episodes/generation for fitness evaluation.\n")
        eval_genomes = make_eval_function(all_episodes, EPISODES_PER_GENERATION)
        winner = pop.run(eval_genomes, n=args.generations)

    with open(best_genome_path, "wb") as f:
        pickle.dump(winner, f)

    # Save fitness history
    for history_file in set([fitness_history_path, os.path.join(RESULTS_DIR, "fitness_history.csv")]):
        with open(history_file, "w") as f:
            f.write("generation,best_fitness,avg_fitness\n")
            for gen, (best, avg) in enumerate(zip(stats.get_fitness_stat(max), stats.get_fitness_mean())):
                f.write(f"{gen},{best},{avg}\n")

    print(f"\nDone. Best genome saved to {best_genome_path}")
    print(f"Best fitness achieved: {winner.fitness:.3f}")


if __name__ == "__main__":
    main()
