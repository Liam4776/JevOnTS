"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'RacketSports',
 'slug': 'racketsports',
 'prompt': 'Classify the RacketSports sample from its complete multivariate time series. Each '
           'input dimension is one synchronized, ordered measurement trace over the same time '
           'steps. Use all channels and their temporal evolution for this task: Six inertial '
           'channels from racket-sport strokes: Badminton Smash, Badminton Clear, Squash '
           'Forehand Boast, or Squash Backhand Boast. Choose one supplied class label.',
 'label_names': {'Badminton_Smash': 'Badminton Smash',
                 'Badminton_Clear': 'Badminton Clear',
                 'Squash_ForehandBoast': 'Squash Forehand Boast',
                 'Squash_BackhandBoast': 'Squash Backhand Boast'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
