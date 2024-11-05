    # Determine simulated accuracy
    accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
    print(accuracy)