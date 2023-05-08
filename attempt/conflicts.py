
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
            if kwargs.attend_for is not None:
                assert kwargs.num_source_prompts > 0 or kwargs.use_private_prompts or kwargs.source_prompts or kwargs.use_prompt_set, "attend for needs source prompts"
                assert model_args.attn_method != "rb", " Attend_for (not None) is for sub method"
            if kwargs.learn_privates:
                assert kwargs.use_private_prompts and model_args.attn_method == "rb", "Use private prompts must be set"
            if kwargs.load_source_prompts:
                assert model_args.attn_tuning, " load source prompts works for attention tuninng"
            if kwargs.num_source_prompts > 0:
                assert model_args.attn_tuning, " This option works for attention tuninng"
            if model_args.add_target is True:
                assert model_args.attn_tuning, " This option works for attention tuninng"
                assert kwargs.num_source_prompts > 0 or kwargs.use_private_prompts or kwargs.source_prompts or kwargs.use_prompt_set, "add target needs source prompts"
            if model_args.attend_target is True:
                assert model_args.attn_tuning, " attend target True is for attention tuninng"
                assert kwargs.num_source_prompts > 0 or kwargs.use_private_prompts or kwargs.source_prompts or kwargs.use_prompt_set, "add target needs source prompts"
            if model_args.attend_input is True:
                assert model_args.attn_tuning, " Attend input is  for attention tuninng"
            if model_args.target_share is None:
                assert not model_args.add_target, " Target share None is for not add_target"
            if model_args.target_share is not None:
                assert model_args.attn_tuning, " This option works for attention tuninng"
                assert model_args.add_target, " Target share not None is for add target"
            if kwargs.use_private_prompts is True:
                assert model_args.attn_tuning, " This option works for attention tuninng"
        elif adapter_args.train_task_adapters:
            pass
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
