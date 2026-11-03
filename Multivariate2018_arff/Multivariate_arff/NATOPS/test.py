"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'NATOPS',
 'slug': 'natops',
 'prompt': 'Classify the NATOPS sample from its complete multivariate time series. Each input '
           'dimension is one synchronized, ordered measurement trace over the same time steps. '
           'Use all channels and their temporal evolution for this task: Motion sequences for '
           'six naval aircraft handling gestures. IDs 1-6 mean I have command, All clear, Not '
           'clear, Spread wings, Fold wings, and Lock wings. Choose one supplied class label.',
 'label_names': {'1.0': 'I have command',
                 '2.0': 'All clear',
                 '3.0': 'Not clear',
                 '4.0': 'Spread wings',
                 '5.0': 'Fold wings',
                 '6.0': 'Lock wings'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
