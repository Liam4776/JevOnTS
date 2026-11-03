"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'FingerMovements',
 'slug': 'fingermovements',
 'prompt': 'Classify the FingerMovements sample from its complete multivariate time series. '
           'Each input dimension is one synchronized, ordered measurement trace over the same '
           'time steps. Use all channels and their temporal evolution for this task: '
           'Twenty-eight EEG channels recorded during finger movement trials; choose left or '
           'right movement. Choose one supplied class label.',
 'label_names': {'left': 'Left', 'right': 'Right'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
