from .full import FullModel
from .incompressible import IncompressibleModel
from .reduced import ReducedModel
from .instant import InstantNucleationModel

# Put all models that can be run in this dictionary
MODEL_OPTIONS = {
    "full": FullModel,
    "incompressible": IncompressibleModel,
    "reduced": ReducedModel,
    "instant": InstantNucleationModel,
}
