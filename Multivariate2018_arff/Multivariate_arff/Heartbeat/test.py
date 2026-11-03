"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'Heartbeat',
 'slug': 'heartbeat',
 'prompt': 'Classify the Heartbeat sample from its complete multivariate time series. Each '
           'input dimension is one synchronized, ordered measurement trace over the same time '
           'steps. Use all channels and their temporal evolution for this task: Multichannel '
           'heartbeat recordings grouped into normal recordings from healthy subjects and '
           'abnormal recordings from patients with cardiac diagnoses. Choose one supplied '
           'class label.',
 'label_names': {'normal': 'Normal', 'abnormal': 'Abnormal'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
