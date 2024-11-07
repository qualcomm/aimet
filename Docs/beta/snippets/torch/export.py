    # Export the model
    # Export the model which saves pytorch model without any simulation nodes and saves encodings file for both
    # activations and parameters in JSON format
    model.export(path='./', filename_prefix='<name_prefix>', dummy_input=dummy_input.cpu())
