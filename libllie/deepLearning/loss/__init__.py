"""Loss registry exports for_teach libllie deep-learning models."""

from .BaseLoss import BaseLoss, L1Loss, MSELoss, SmoothL1Loss, CharbonnierLoss
from .ZeroDCE_Loss import ZeroDCE_Loss, ZeroDCE_extension_Loss
from .Sci_Loss import Sci_Loss
from .RUAS_Loss import RUAS_Loss, RUASEnhanceLoss, RUASDenoiseLoss
from .LEDNet_Loss import LEDNet_Loss, LEDNetLoss, VGG19PerceptualLoss
from .DarkIR_Loss import DarkIR_Loss, DarkIRLoss, DarkIREdgeLoss, DarkIRVGGFeatureLoss
from .ZeroIG_Loss import ZeroIG_Loss, ZeroIGLoss
from .URetinex_Loss import URetinex_Loss, URetinexLoss
from .RetinexFormer_Loss import RetinexFormer_Loss, RetinexFormerLoss
from .LLNet_Loss import LLNet_Loss, LLNetLoss
from .KinD_Loss import KinD_Loss, KinDLoss
from .KinDPlusPlus_Loss import KinDPlusPlus_Loss, KinDPlusPlusLoss
from .EnlightenGAN_Loss import EnlightenGAN_Loss, EnlightenGANLoss
from .LLFlow_Loss import LLFlow_Loss, LLFlowLoss


__all__ = [
    'BaseLoss',
    'L1Loss',
    'MSELoss',
    'SmoothL1Loss',
    'CharbonnierLoss',
    'ZeroDCE_Loss',
    'ZeroDCE_extension_Loss',
    'Sci_Loss',
    'RUAS_Loss',
    'LEDNet_Loss',
    'LEDNetLoss',
    'VGG19PerceptualLoss',
    'DarkIR_Loss',
    'DarkIRLoss',
    'DarkIREdgeLoss',
    'DarkIRVGGFeatureLoss',
    'ZeroIG_Loss',
    'ZeroIGLoss',
    'URetinex_Loss',
    'URetinexLoss',
    'RetinexFormer_Loss',
    'RetinexFormerLoss',
    'LLNet_Loss',
    'LLNetLoss',
    'KinD_Loss',
    'KinDLoss',
    'KinDPlusPlus_Loss',
    'KinDPlusPlusLoss',
    'EnlightenGAN_Loss',
    'EnlightenGANLoss',
    'LLFlow_Loss',
    'LLFlowLoss',
]
