"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'PenDigits',
 'slug': 'pendigits',
 'prompt': 'Classify the PenDigits sample from its complete multivariate time series. Each '
           'input dimension is one synchronized, ordered measurement trace over the same time '
           'steps. Use all channels and their temporal evolution for this task: '
           'Two-dimensional pen coordinates sampled at eight points from a handwritten digit; '
           'choose digit 0-9. Choose one supplied class label.',
 'label_names': {'0': 'Digit 0',
                 '1': 'Digit 1',
                 '2': 'Digit 2',
                 '3': 'Digit 3',
                 '4': 'Digit 4',
                 '5': 'Digit 5',
                 '6': 'Digit 6',
                 '7': 'Digit 7',
                 '8': 'Digit 8',
                 '9': 'Digit 9'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
