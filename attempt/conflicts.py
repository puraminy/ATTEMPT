
####### check conflicts of options
def check_conflicts(model_args, data_args, training_args, adapter_args, kwargs):
    n_tasks = len(data_args.task_name)
    trainer_shuffle = kwargs.setdefault("trainer_shuffle", False)
    if n_tasks > 1:
        assert not trainer_shuffle, "Trainer can't be shuffled for multi-task. The data is interleaved"

    if adapter_args.prefix_tuning:
        assert not adapter_args.prompt_tuning, "Prompt tuning and prefix tuning can't be on at the same time"
        if training_args.do_train:
            if model_args.attn_learning_rate is not None:
                assert kwargs.use_optimizer, "Attention learning uses optimizer" 
            else:
                assert not kwargs.use_optimizer, "No need to use optimizer" 
            assert training_args.learning_rate > 0.01, "Learning rate is too small for prefix tuning"
        if not model_args.attn_prefix_tuning:
            pass

    elif not adapter_args.prompt_tuning:
        assert not adapter_args.prefix_tuning, "Prompt tuning and prefix tuning can't be on at the same time"
        if training_args.do_train:
            assert training_args.learning_rate < 0.01, "Learning rate is too high for fine tuning"
            assert kwargs.opt_type == "regular", "Fine tuning needs regular optimizer"
