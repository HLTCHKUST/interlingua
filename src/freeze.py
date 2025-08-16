def freeze_update(model, n_first = 7, embed = True, final = True):

    frozen_layer_ids = range(n_first)
    print("Freezing")
    for layer_id in frozen_layer_ids:
        for param in model.model.layers[layer_id].parameters():
            param.requires_grad = False
    
    if embed:
        # 0 in model_args.frozen_layer_ids
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False
    
    if final:
        # len(model.model.layers)-1 in model_args.frozen_layer_ids
        for param in model.lm_head.parameters():
            param.requires_grad = False
        for param in model.model.norm.parameters():
            param.requires_grad = False

    print("Done")