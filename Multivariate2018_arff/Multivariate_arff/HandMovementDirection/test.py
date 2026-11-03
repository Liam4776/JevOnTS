"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'HandMovementDirection',
 'slug': 'handmovementdirection',
 'prompt': 'Classify the HandMovementDirection sample from its complete multivariate time '
           'series. Each input dimension is one synchronized, ordered measurement trace over '
           'the same time steps. Use all channels and their temporal evolution for this task: '
           'Ten MEG channels recorded during joystick movement in one of four directions: '
           'right, forward, left, or backward. Choose one supplied class label.',
 'label_names': {'right': 'Right',
                 'forward': 'Forward',
                 'left': 'Left',
                 'backward': 'Backward'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
