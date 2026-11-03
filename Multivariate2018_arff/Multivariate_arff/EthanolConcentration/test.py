"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'EthanolConcentration',
 'slug': 'ethanolconcentration',
 'prompt': 'Classify the EthanolConcentration sample from its complete multivariate time '
           'series. Each input dimension is one synchronized, ordered measurement trace over '
           'the same time steps. Use all channels and their temporal evolution for this task: '
           'Raw spectra from whisky bottles; E35, E38, E40, and E45 represent ethanol '
           'concentrations of 35%, 38%, 40%, and 45%. Choose one supplied class label.',
 'label_names': {'E35': '35% ethanol',
                 'E38': '38% ethanol',
                 'E40': '40% ethanol',
                 'E45': '45% ethanol'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
