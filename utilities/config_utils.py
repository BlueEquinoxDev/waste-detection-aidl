from enum import auto, Enum

class DatasetType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    
class LoggerType(Enum):
    TENSORBOARD = auto()
    WANDB = auto()
    
class TaskType(Enum):
    SEGMENTATION = auto()
    CLASSIFICATION = auto()

class ClassificationCategoryType(Enum):
    CATEGORY = auto()
    SUPERCATEGORY = auto()
    CUSTOM = auto()