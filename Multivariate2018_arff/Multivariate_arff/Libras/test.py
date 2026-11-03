"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'Libras',
 'slug': 'libras',
 'prompt': 'Classify the Libras sample from its complete multivariate time series. Each input '
           'dimension is one synchronized, ordered measurement trace over the same time steps. '
           'Use all channels and their temporal evolution for this task: Two-dimensional hand '
           'movement trajectories grouped into 15 gesture classes with numeric IDs. Choose one '
           'supplied class label.',
 'label_names': {}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
