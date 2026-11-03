"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'SelfRegulationSCP1',
 'slug': 'selfregulationscp1',
 'prompt': 'Classify the SelfRegulationSCP1 sample from its complete multivariate time series. '
           'Each input dimension is one synchronized, ordered measurement trace over the same '
           'time steps. Use all channels and their temporal evolution for this task: Six EEG '
           'channels from a self-regulation experiment; classes are negativity and positivity. '
           'Choose one supplied class label.',
 'label_names': {'negativity': 'Negativity', 'positivity': 'Positivity'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
