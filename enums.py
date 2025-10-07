from enum import Enum, auto

class MappingEnum(Enum):
    SYMMETRIC = auto()
    ASYMMETRIC = auto()

class GranularityEnum(Enum):
    PER_ROW = auto()
    # PER_COLUMN = auto()
    PER_TENSOR = auto()