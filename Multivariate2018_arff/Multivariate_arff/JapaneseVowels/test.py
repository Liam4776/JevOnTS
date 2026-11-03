"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'JapaneseVowels',
 'slug': 'japanesevowels',
 'prompt': 'Classify the JapaneseVowels sample from its complete multivariate time series. '
           'Each input dimension is one synchronized, ordered measurement trace over the same '
           'time steps. Use all channels and their temporal evolution for this task: '
           'Twelve-dimensional spectral trajectories from utterances of Japanese vowels a and '
           'e; the label identifies one of nine speakers. Choose one supplied class label.',
 'label_names': {'1': 'Speaker 1',
                 '2': 'Speaker 2',
                 '3': 'Speaker 3',
                 '4': 'Speaker 4',
                 '5': 'Speaker 5',
                 '6': 'Speaker 6',
                 '7': 'Speaker 7',
                 '8': 'Speaker 8',
                 '9': 'Speaker 9'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
