
####### check conflicts of options
def check_conflicts(model_args, data_args, training_args, adapter_args, kwargs):
    resolved = False
    try:
        n_tasks = len(data_args.task_name)
        trainer_shuffle = kwargs.setdefault("trainer_shuffle", False)
        if False: #TODO ignored n_tasks > 1:
            kwargs.trainer_shuffle = False
            resolved = True
            assert not trainer_shuffle, "Trainer can't be shuffled for multi-task. The data is interleaved"

        if adapter_args.prefix_tuning:
            assert not adapter_args.prompt_tuning, "Prompt tuning and prefix tuning can't be on at the same time"
            if training_args.do_train:
                if model_args.attn_learning_rate is not None:
                    assert kwargs.use_optimizer, "Attention learning uses optimizer" 
                else:
                    assert not kwargs.use_optimizer, "No need to use optimizer" 
                assert training_args.learning_rate > 0.01, "Learning rate is too small for prefix tuning"
            if not model_args.attn_tuning:
                if model_args.attend_target is False:
                    assert model_args.add_target is True, "Can't both attend target and add target be false"

        elif adapter_args.prompt_tuning:
            if model_args.add_target is True:
                assert model_args.attn_tuning, " This option works for attention tuninng"
            if model_args.attend_target is True:
                assert model_args.attn_tuning, " This option works for attention tuninng"
            if model_args.attend_input is True:
                assert model_args.attn_tuning, " This option works for attention tuninng"
            if model_args.target_share >= 0:
                assert model_args.attn_tuning, " This option works for attention tuninng"
            if kwargs.use_private_prompts is True:
                assert model_args.attn_tuning, " This option works for attention tuninng"
        else:
            assert not adapter_args.prefix_tuning, "Prompt tuning and prefix tuning can't be on at the same time"
            if training_args.do_train:
                assert training_args.learning_rate < 0.01, "Learning rate is too high for fine tuning"
                assert kwargs.opt_type == "regular", "Fine tuning needs regular optimizer"

            if model_args.target_share is not None:
                assert model_args.add_target is True, "Target share needs target to be added"
    except AssertionError as e:
        msg = str(e.args[0])
        if resolved: msg = "Resolved:" + msg 
        else: msg = "Conflict:" + msg
        return resolved, msg 
    return True, ""
