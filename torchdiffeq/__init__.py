# Hack to allow for training on Mac (MPS) which supports only float32
import torch
if torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_dtype(torch.float64)

from ._impl import odeint
from ._impl import odeint_adjoint
from ._impl import odeint_event
from ._impl import odeint_dense
__version__ = "0.2.4"
