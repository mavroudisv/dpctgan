from ctgan.synthesizers.ctgan import CTGANSynthesizer
from ctgan.synthesizers.dpctgan import dpCTGANSynthesizer
from ctgan.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'dpCTGANSynthesizer',
	'CTGANSynthesizer',
    'TVAESynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
