"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'MotorImagery',
 'slug': 'motorimagery',
 'prompt': 'Classify the MotorImagery sample from its complete multivariate time series. Each '
           'input dimension is one synchronized, ordered measurement trace over the same time '
           'steps. Use all channels and their temporal evolution for this task: Sixty-four EEG '
           'channels from a brain-computer interface task; classes are finger or tongue motor '
           'imagery. Choose one supplied class label.',
 'label_names': {'finger': 'Finger imagery', 'tongue': 'Tongue imagery'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
