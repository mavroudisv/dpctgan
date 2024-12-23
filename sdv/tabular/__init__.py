"""Models for tabular data."""

from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, TVAE
from sdv.tabular.dpctgan import dpCTGAN

__all__ = (
    'GaussianCopula',
    'CTGAN',
    'dpCTGAN',
    'TVAE',
    'CopulaGAN',
)
