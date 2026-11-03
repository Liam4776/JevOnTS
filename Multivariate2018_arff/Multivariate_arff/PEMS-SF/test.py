"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'PEMS-SF',
 'slug': 'pems_sf',
 'prompt': 'Classify the PEMS-SF sample from its complete multivariate time series. Each input '
           'dimension is one synchronized, ordered measurement trace over the same time steps. '
           'Use all channels and their temporal evolution for this task: Traffic occupancy '
           'from 963 San Francisco freeway sensors sampled every 10 minutes for 144 time '
           'steps. Labels 1-7 correspond to Monday-Sunday. Choose one supplied class label.',
 'label_names': {'1.0': 'Monday',
                 '2.0': 'Tuesday',
                 '3.0': 'Wednesday',
                 '4.0': 'Thursday',
                 '5.0': 'Friday',
                 '6.0': 'Saturday',
                 '7.0': 'Sunday'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
