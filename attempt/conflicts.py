
####### check conflicts of options
def check_conflicts(model_args, data_args, training_args, adapter_args, kwargs):
    n_tasks = len(data_args.task_name)
    method = kwargs.setdefault("method", [])
    trainer_shuffle = kwargs.setdefault("trainer_shuffle", False)
    if not adapter_args.prompt_tuning:
        assert "sup" in data_args.template, "The template must be one of sup or unsup methods"
    if n_tasks > 1:
        assert not trainer_shuffle, "Trainer can't be shuffled for multi-task. The data is interleaved"
    if adapter_args.prompt_tuning:
        assert "pt" in method, "Prompt tuning is not in the selected methods"
        assert "-pt" in data_args.template, "Prompt tuning is not supported by the selected template"
        assert kwargs.use_optimizer, "Prompt tuning uses optimizer" 
        assert not adapter_args.prefix_tuning, "Prompt tuning and prefix tuning can't be both on" 
    else:
        assert not "-pt" in data_args.template, "The selected template requires prompt tuning"
        if model_args.attn_learning_rate is not None:
            assert kwargs.use_optimizer, "Attention learning uses optimizer" 
        else:
            assert not kwargs.use_optimizer, "No need to use optimizer" 
        if adapter_args.prefix_tuning:
            assert "prt" in method, "prefix tuning is not in the selected methods"
            assert training_args.learning_rate > 0.01, "Learning rate is too small for prefix tuning"
        else:
            assert training_args.learning_rate < 0.01, "Learning rate is too high for fine tuning"
            assert "ft" in method, "fine tuning is not in the selected methods"
            assert kwargs.opt_type == "regular", "Fine tuning needs regular optimizer"
            assert adapter_args.prompt_learning_rate is None, "For fine tuning prompt learning rate isn't used" 
