def selective_grad_freeze(model, n_first = 8, embed = True, final = True):
    """
    Selectively freezing trainable parameters.
    Default is to freeze the first 8 layers, the token embedding, final layer normalization, 
    and the language modeling head (output projection layer).
    """

    frozen_layer_ids = range(n_first+1)
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