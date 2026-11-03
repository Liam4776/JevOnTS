"""Run Jev on this dataset's canonical multivariate test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_multivariate_runner import run

CONFIG = {'task': 'FaceDetection',
 'slug': 'facedetection',
 'prompt': 'Classify the FaceDetection sample from its complete multivariate time series. Each '
           'input dimension is one synchronized, ordered measurement trace over the same time '
           'steps. Use all channels and their temporal evolution for this task: MEG brain '
           'recordings for Face Detection: class 1 is Face and class 0 is Scramble. Choose one '
           'supplied class label.',
 'label_names': {'0': 'Scramble', '1': 'Face'}}

if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
