"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'PhonemeSpectra',
 'slug': 'phonemespectra',
 'prompt': 'Classify the PhonemeSpectra sample from its complete multivariate time series. '
           'Each input dimension is one synchronized, ordered measurement trace over the same '
           'time steps. Use all channels and their temporal evolution for this task: Eleven '
           'spectral channels for phonetic time series; labels are 39 phoneme symbols. Choose '
           'one supplied class label.',
 'label_names': {}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
