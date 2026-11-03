"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'InsectWingbeat',
 'slug': 'insectwingbeat',
 'prompt': 'Classify the InsectWingbeat sample from its complete multivariate time series. '
           'Each input dimension is one synchronized, ordered measurement trace over the same '
           'time steps. Use all channels and their temporal evolution for this task: Audio '
           'spectrogram sequences for 10 insect groups, including male and female mosquitoes '
           'and flies; labels identify group and sex. Choose one supplied class label.',
 'label_names': {'Aedes_female': 'Aedes female',
                 'Aedes_male': 'Aedes male',
                 'Fruit_flies': 'Fruit flies',
                 'House_flies': 'House flies',
                 'Quinx_female': 'Quinx female',
                 'Quinx_male': 'Quinx male',
                 'Stigma_female': 'Stigma female',
                 'Stigma_male': 'Stigma male',
                 'Tarsalis_female': 'Tarsalis female',
                 'Tarsalis_male': 'Tarsalis male'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
