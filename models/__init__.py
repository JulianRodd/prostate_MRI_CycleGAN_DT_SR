import importlib
import sys
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    Args:
        model_name (str): Name of the model to find

    Returns:
        class: The model class if found

    Raises:
        ImportError: If the model module cannot be imported
        RuntimeError: If the model class cannot be found in the module
    """
    try:
        # Construct the module path
        model_filename = "models." + model_name + "_model"

        # If the module is already imported, reload it to ensure fresh state
        if model_filename in sys.modules:
            module = sys.modules[model_filename]
            modellib = importlib.reload(module)
        else:
            modellib = importlib.import_module(model_filename)

        # Find the model class
        target_model_name = model_name.replace("_", "") + "model"
        model = None

        for name, cls in modellib.__dict__.items():
            if (
                name.lower() == target_model_name.lower()
                and isinstance(cls, type)
                and issubclass(cls, BaseModel)
                and cls != BaseModel
            ):
                model = cls
                break

        if model is None:
            raise RuntimeError(
                f"In {model_filename}.py, there should be a subclass of BaseModel "
                f"with class name that matches {target_model_name} in lowercase."
            )

        return model

    except ImportError as e:
        print(f"Error importing model {model_name}: {str(e)}")
        raise
    except Exception as e:
        print(f"Error finding model {model_name}: {str(e)}")
        raise


def get_option_setter(model_name):
    """Get the static method modify_commandline_options from model class.

    Args:
        model_name (str): Name of the model

    Returns:
        function: The option setter function
    """
    try:
        model_class = find_model_using_name(model_name)
        if not hasattr(model_class, "modify_commandline_options"):
            raise AttributeError(
                f"Model class {model_class.__name__} does not implement "
                "modify_commandline_options"
            )
        return model_class.modify_commandline_options
    except Exception as e:
        print(f"Error getting option setter for {model_name}: {str(e)}")
        raise


def create_model(opt):
    """Create a model instance.

    Args:
        opt: Command line options

    Returns:
        BaseModel: An instance of the model
    """
    try:
        # Ensure required options are present
        if not hasattr(opt, "model"):
            raise ValueError("Options must include 'model' attribute")

        # Find and instantiate the model
        model_class = find_model_using_name(opt.model)
        instance = model_class()

        # Initialize the model
        if not hasattr(instance, "initialize"):
            raise AttributeError(
                f"Model class {model_class.__name__} does not implement initialize"
            )
        instance.initialize(opt)

        print(f"Model [{instance.name()}] was created")
        return instance

    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise
