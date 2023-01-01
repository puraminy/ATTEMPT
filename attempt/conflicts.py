
####### check conflicts of options
def check_conflicts(model_args, data_args, training_args, adapter_args):
    if adapter_args.prompt_tuning:
        assert model_args.use_optimizer, "Prompt tuning uses optimizer" 
        assert not adapter_args.prefix_tuning, "Prompt tuning and prefix tuning can't be both on" 
    else:
        if model_args.attn_learning_rate is not None:
            assert model_args.use_optimizer, "Attention learning uses optimizer" 
        else:
            assert not model_args.use_optimizer, "No need to use optimizer" 

