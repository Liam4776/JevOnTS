"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'BasicMotions',
 'slug': 'basicmotions',
 'prompt': 'Classify the BasicMotions sample from its complete multivariate time series. Each '
           'input dimension is one synchronized, ordered measurement trace over the same time '
           'steps. Use all channels and their temporal evolution for this task: Six smartwatch '
           'channels: three accelerometer and three gyroscope axes. Classes are Standing, '
           'Running, Walking, and Badminton. Choose one supplied class label.',
 'label_names': {'Standing': 'Standing',
                 'Running': 'Running',
                 'Walking': 'Walking',
                 'Badminton': 'Badminton'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
