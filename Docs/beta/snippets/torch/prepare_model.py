# Prepare the model
from aimet_torch.model_preparer import prepare_model
prepared_model = prepare_model(<model>)

# where <model> is a torch.nn.Module