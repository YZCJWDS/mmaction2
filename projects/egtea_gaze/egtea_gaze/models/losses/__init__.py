"""Loss exports for the EGTEA gaze project."""

from .gaze_losses import GazeBCELoss, GazeKLLoss, GazeMSELoss

__all__ = ['GazeBCELoss', 'GazeKLLoss', 'GazeMSELoss']

