from .full import FullModel
from .incompressible import IncompressibleModel
from .ideal import IdealModel
from .reduced import ReducedModel
from .instant import InstantNucleationModel

# Put all models that can be run in this dictionary
MODEL_OPTIONS = {
    "full": FullModel,
    "incompressible": IncompressibleModel,
    "ideal": IdealModel,
    "reduced": ReducedModel,
    "instant": InstantNucleationModel,
}
