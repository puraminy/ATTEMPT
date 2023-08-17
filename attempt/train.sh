#!/usr/bin/bash
# shopt -s expand_aliases 
# source ~/aa

extra_params=""
run_params=""
bash_params=""
main_vars=""
for i in $@
do
   case $i in
      --*) 
            q=${i#"--"}
            extra_params="${extra_params} --@${q}"; 
            p=${i%=*}
            main_vars="${main_vars}${p}"
            g=extra
       ;;
      _*) bash_params="${bash_params} $i"; g=bash;;
      -*) run_params="${run_params} $i"; g=run;;
       # Other Parameter 
       *) p=$i
       if [ "$g" = run ]
       then
          run_params="${run_params} $p"
          g=0
       elif [ "$g" = bash ]; then
          bash_params=${bash_params%=*}
          bash_params="${bash_params}=$i" 
          g=arr
       elif [ "$g" = arr ]; then
          bash_params="${bash_params}#$i" 
          g=arr
       else
          others="$others $p"
       fi
       ;;
 esac
 if [ "$g" = bash ]; then
    bash_params="${bash_params}=True";
 fi
done
main_vars=${main_vars#"--"}
main_vars="${main_vars}--task_name"

################## Utils
get_com_ppx() {
   arr=($(echo -n $1 | sed "s/${sep}/ /g"))
   IFS=$'\n' sorted=($(sort <<<"${arr[*]}")); unset IFS
   tlist=$(printf "%s-" "${sorted[@]}")
   tlist=${tlist%-}
   result=${tlist}
}

################

echo "==================== Train.sh ======================"
echo "Main experiment variables: $main_vars"
echo "Bash Prarams: ${bash_params}"
echo "Extra Prarams: ${extra_params}"
echo "Run Prarams: ${run_params}"
eval "${bash_params}"
echo "Tasks: $_tasks"

if [ -n "$_ttasks" ]; then
   _tasks="${_tasks}#qnli#rte#mrpc#qqp"
fi
if [ -n "$_gtasks" ]; then
   _tasks="${_tasks}#cola#mnli#qnli#rte#qqp#mrpc#sst2#stsb"
fi
if [ -n "$_ltasks" ]; then
   _tasks="mnli#wnli#paws#mrpc#imdb#sst2"
fi

if [ -n "$_otasks" ]; then
   _tasks="${_tasks}#multinli#piqa#newsqa#searchqa#triviaqa#nq#hotpotqa#social_i_qa#commonsense_qa#winogrande#scitail#yelp_polarity#tweet-eval#imdb"
fi
sgtasks="superglue-wsc.fixed#superglue-wic#superglue-boolq#superglue-cb#superglue-rte#superglue-copa"
if [ -n "$_sgtasks" ]; then
   _tasks="${_tasks}#${sgtasks}"
fi
if [ -n "$_atasks" ]; then
   _tasks="${_tasks}#xAttr#xIntent#xReact#oReact#oEffect#oWant#xWant#xEffect#xNeed"
fi
if [ -n "$_satasks" ]; then
   _tasks="${_tasks}#xAttr#xIntent#xReact#xWant#oWant"
fi
if [ -n "$_ltasks2" ]; then
   _tasks="mnli#qnli#qqp#mrpc#imdb#sst2#superglue-boolq#stsb"
fi

if [ -n "$_sstasks" ]; then
   _stasks="mnli#${sgtasks}"
fi
if [ -n "$_spt" ]; then
 _nsp=${#_tasks[@]}
fi
if [ -z "$_pat" ] && [ -z "$_ft" ] && [ -z "$_pt" ]; then
 _pat=True
fi
if [ -n "$_pat" ]; then
   if [ -n "$_seqt" ] && [ -z "$_stasks" ]; then
      _stasks=$_tasks
   fi
   if [ -z "$_nsp" ] && [ -z "$_stasks" ]; then
      echo "_stasks (source tasks) is missinge e.g. _stasks mnli qqp rte "
      exit
   fi

   if [ -z "$_nsp" ] && [ -z "$_ppx" ]; then
      if [ -z "$_lsp" ] || [ "$_lsp" = "True" ]; then
         echo "_ppx (prompts prefix) is missinge e.g. _ppx nli "
         exit
      fi
   fi
   if [ -z "$_src" ]; then
      stasks=$_stasks
      echo "Source tasks are: $stasks"
      arr=($(echo -n $_stasks | sed "s/\#/ /g"))
      _src=""
      for t in "${arr[@]}"; do
         _src="${_src}@$t"
      done
      # _src=${_src#"@"}
      echo "Used source prompts are: ${_src}"
   fi
fi

echo "Tasks: ===================="
echo $_tasks

model=t5-base
tokname=t5-base
home=$HOME
case "$HOME" in 
   *root*)
      # Colab 
      home=/content/drive/MyDrive
      ;;
   *ahmad*)
      # Do stuff
      model=t5-base
      tokname=t5-base
      ;;
esac
if [ -n "$_base" ]; then
   model=t5-base
   tokname=t5-base
fi
if [ -n "$_mt5" ]; then
   model=mt5-base
   tokname=mt5-base
fi
if [ -n "$_pars" ]; then
   model=parsT5-base
   tokname=parsT5-base
fi
if [ -n "$_lmb" ]; then
   # model=google/t5-v1_1-base
   model=t5-lmb
   tokname=t5-base
fi
if [ -n "$_v1" ]; then
   # model=google/t5-v1_1-base
   model=t5-v1
   tokname=t5-base
fi
if [ -n "$_large" ]; then
   model=t5-large
   tokname=t5-large
fi
if [ -n "$_max" ]; then
   _msl=450
   _mtl=25
fi
if [ -n "$_self" ]; then
   _exp=self
fi
if [ -n "$_unsup" ]; then 
   _temp=unsup-p0-px0-px-pt
fi
if [ -n "$_pt" ]; then 
    _met="pt"
fi
if [ -n "$_ft" ]; then 
    _met="ft"
fi
if [ -n "$_all" ]; then 
   _tn=-1
   _vn=20
   _tsn=-1 
fi
if [ -n "$_all_test" ]; then 
   _tsn=-1 
fi
if [ -n "$_eval" ]; then
   _dotest=True
   _doeval=False
   _train=False
   _lp=True
fi

if [ -z $_tasks ]; then
   echo "_tasks is missinge (target task or tasks) e.g. _tasks mnli rte "
   exit 1
fi
_tasks=${_tasks##\#}
echo "---------- Tasks: $_tasks"
_task=$_tasks

if [ -z "$_single" ] && [ -z "$_multi" ]; then
   if [ -z "$_pt" ]; then
      _multi=True
   fi
fi

if [ -n "$_multi" ]; then
   _task=$(echo "$_tasks" | sed "s/\#/@/g")
   echo "Multi Tasks: $_task"
fi

if [ -n "$_pvtasks" ]; then
   exit
fi
case $run_params in
   *"-d"*) # debug is enabled
      _test=True
      ;;
   *"-nd"*) # debug is enabled
      run_params=$(echo "$run_params" | sed "s/-nd/-d/")
      ;;
esac
# wrap experiments
folder=${PWD##*/}          
log=${home}/logs   
echo "log: ${log}"

if [ -z "$_bs" ]; then  _bs=8; fi
if [ -z "$_lr" ]; then  _lr=0.05; fi
if [ -z "$_alr" ]; then _alr=0.1; fi
if [ -z "$_adir" ]; then  _adir=-1; fi
if [ -z "$_tmpr" ]; then  _tmpr=5.; fi
if [ -z "$_inp" ]; then  _inp=False; fi
if [ -z "$_numt" ]; then  _numt=50; fi
if [ -z "$_pl" ]; then  _pl=$_numt; fi
if [ -z "$_sr" ]; then  _sr=False; fi # save router
if [ -z "$_usr" ]; then  _usr=False; fi # use saved router
if [ -z "$_upp" ]; then  _upp=False; fi # use private prompts 
if [ -z "$_lpp" ]; then  _lpp=False; fi # load private prompts 
if [ -z "$_addt" ]; then  _addt=False; fi # Add Target 
if [ -z "$_nsp" ]; then  
   _nsp=0; 
elif [ -z "$_lsp" ]; then
   _lsp=False
fi
if [ -z "$_learn_sp" ]; then  _learn_sp=True; fi
if [ -z "$_addt" ]; then  _addt=False; fi
if [ -z "$_attn" ]; then  _attn=rb; fi
if [ -z "$_cmm" ]; then  _cmm="wavg#cat"; fi
if [ -z "$_at" ]; then  _at=True; fi
if [ -z "$_sp" ]; then  _sp=True; fi # save prompts
if [ -z "$_ai" ]; then  _ai=False; fi
if [ -z "$_lp" ]; then  _lp=True; fi
if [ -z "$_pdir" ]; then  _pdir=prompts; fi
if [ -z "$_msl" ]; then  _msl=400; fi
if [ -z "$_mtl" ]; then  _mtl=50; fi
if [ -z "$_sph" ]; then  _sph=1; fi
if [ -z "$_mc" ]; then  _mc=False; fi #multi choice format

# eeeee
if [ -n "$_test" ]; then
   _rem=True
   if [ -z "$_tn" ]; then  _tn=64; fi
   _vn=2
   _tsn=2 
   _ep=1
   _doeval=False
fi
if [ -z "$_train" ]; then  _train=True; fi
if [ -z "$_doeval" ]; then  _doeval=False; fi
if [ -z "$_dotest" ]; then  _dotest=True; fi
if [ -z "$_tn" ]; then  
   _tn=500; 
else
   main_vars="${main_vars}--max_train_samples"
fi
if [ -z "$_vn" ]; then  _vn=50; fi
if [ -z "$_tsn" ]; then _tsn=500; fi
if [ -z "$_ep" ]; then  
   _ep=20; 
else
   main_vars="${main_vars}--num_train_epochs"
fi
if [ -z "$_err" ]; then  _err=break; fi
onError=$_err
params=""


if [ -z "$_rels" ]; then  _rels="none"; fi
if [ -z "$_met" ]; then
   _met="ptat"
fi
if [ "$_model" = "path" ]; then
   params="${params} --model_name_or_path=~${PWD}/trial=1"
fi

if [ "$_train" = "False" ]; then
   _tn=0
fi
if [ "$main_vars" != "" ]; then
   run_params="${run_params} -mv ${main_vars}"
fi
for method in $_met; do
   echo "==================method: $method === epochs: $_ep ===== samples: $_train =========="
   if [ -z "$_exp" ]; then 
      if [ -n "$_nsp" ] && [ "$_nsp" != "0" ]; then
         _exp="nsp-$_nsp"
      elif [ -n "$_seqt" ]; then
         if [ -n "$_upp" ]; then
            _exp="seqt-upp"
         else
            _exp="seqt"
         fi
      elif [ -n "$_cat" ]; then
         _exp=$_cat
      else
         _exp=$_task
      fi
      if [ -n "$_eval" ]; then 
         _exp="${_exp}_eval"; 
      fi
      _exp="${_exp}-${_lsp}-${_learn_sp}-${_lpp}"
   fi

params="${params} --@method=$method"

# data  ddddd
params="${params} --data_path=atomic2020"
params="${params} --use_all_data=False"
params="${params} --@max_train_samples=$_tn"
params="${params} --max_val_samples=$_vn"
params="${params} --max_test_samples=$_tsn"
params="${params} --samples_per_head=$_sph"
params="${params} --multi_choice=$_mc"
params="${params} --map_labels=True"
#params="${params} --data_seed=123"
params="${params} --overwrite_cache=True"
if [ -z $_seed ]; then
   params="${params} --@data_seed=123"
else
   params="${params} --@data_seed=$_seed"
fi

# task

params="${params} --@task_name=$_task"
params="${params} --add_prefix=False"
params="${params} --ds_config=en@"
params="${params} --max_source_length=$_msl"
params="${params} --max_target_length=$_mtl"
params="${params} --test_ds_config=full-test@"

if [ "$_exp" = "self" ]; then
   # exp="${PWD#$log/}"
   exp=self
   _train=False
else
if [ -n "$_cat" ]; then
   exp=$_cat/$_exp
else
   exp=$method/$_exp
fi
fi

params="${params} --do_train=$_train"
# operations
params="${params} --do_test=$_dotest"
params="${params} --do_eval=$_doeval"
# Saving
params="${params} --report_to=wandb@"
params="${params} --model_name_or_path=$model"
params="${params} --tokenizer_name=$tokname"
params="${params} --use_fast_tokenizer=True"
params="${params} --save_total_limit=1"
params="${params} --save_checkpoint=False"
params="${params} --save_model=False"
params="${params} --load_best_model_at_end=True" # We handle it by loading best prompts

# training 
params="${params} --per_device_train_batch_size=$_bs"
params="${params} --per_device_eval_batch_size=$_bs"
params="${params} --trainer_shuffle=True"
params="${params} --skip_specials=True"
params="${params} --@num_train_epochs=$_ep"
params="${params} --adjust_epochs=True"
params="${params} --warmup_steps=none"
# generation
params="${params} --num_beams=1"

if [ "$method" = "ft" ]; then
   params="${params} --warmup_steps=500"
   params="${params} --learning_rate=0.0003"
   params="${params} --opt_type=regular"
   params="${params} --per_device_train_batch_size=4"
   if [ -z $_temp ]; then
      params="${params} --@template=unsup#sup"
   else
      params="${params} --@template=$_temp"
   fi
fi

# prefix tuning
# xxxxxxxxxxxx
if [ "$method" = "px" ] || [ "$method" = "at" ]; then
   params="${params} --learning_rate=0.3"
   params="${params} --prefix_tuning=True"
   params="${params} --prefix_dim=100"
   params="${params} --opt_type=regular"
   params="${params} --use_optimizer=False"
   params="${params} --config=base"
   params="${params} --prefix_dir=prefixes"
fi

if [ "$method" = "at" ]; then
   params="${params} --attn_tuning=True"
   params="${params} --config=attempt"
   params="${params} --prompt_embedding_path=xWant.pt@xNeed.pt@xIntent.pt"
   params="${params} --attn_method=sub"
   params="${params} --shared_attn=True"
fi
if [ -z "$_ppx" ]; then  _ppx="${_ep}${_tn}"; fi
if [ -z "$_rpx" ]; then  _rpx="${_ep}${_tn}"; fi
if [ -z "$_opx" ]; then  _opx="pat"; fi

# prompt tuning common settings
if [ "$method" = "pt" ] || [ "$method" = "ptat" ]; then
   params="${params} --prompt_tuning=True"
   params="${params} --use_optimizer=True"
   params="${params} --opt_type=regular"
   params="${params} --prompt_encoders_dir=$_pdir"
   params="${params} --load_prompts=$_lp"
   params="${params} --ignore_train_if_exist=True"
   params="${params} --prompts_prefix=$_ppx"
   params="${params} --output_prompts_prefix=$_opx"
fi

if [ -z $_adapter ]; then 
   echo "no Adapter"
else
   params="${params} --learning_rate=0.0003"
   params="${params} --train_task_adapters=True"
   params="${params} --task_reduction_factor=32"
   params="${params} --unfreeze_lm_head=False"
   params="${params} --unfreeze_layer_norms=True"
   params="${params} --trainer_shuffle=False"
fi 

### pppppppppppppppppppppp
if [ "$method" = "pt" ]; then 
   params="${params} --prompt_learning_rate=$_lr"
   params="${params} --@num_prompt_tokens=$_numt"
   params="${params} --@prompt_encoder_type=mlp#!lstm"
   if [ -z "$_temp" ]; then
      params="${params} --@template=pt"
   else
      params="${params} --@template=$_temp"
   fi
   params="${params} --@num_prompts=2"
   params="${params} --@task_comb=none"
   params="${params} --@prompt_sharing=shared_encoders"
   params="${params} --init_from_words=False"
   params="${params} --load_source_prompts=False"
   params="${params} --load_prompts=$_lp"
   params="${params} --ignore_if_not_exist=True"
   params="${params} --@learn_loaded_prompts=False"
   params="${params} --save_all_prompts=$_sp"
   params="${params} --rels=$_rels"
fi
# aaaaaaaaaaaaaa
if [ "$method" = "ptat" ] || [ "$method" = "adapter" ]; then
   params="${params} --prompt_encoder_type=mlp#!emb#!lstm"
   if [ -z $_lsp ]; then
      params="${params} --@load_source_prompts=True#False"
   else
      params="${params} --@load_source_prompts=$_lsp"
   fi
   params="${params} --@num_prompt_tokens=$_numt"
   params="${params} --@prompt_length=$_pl"
   params="${params} --@num_source_prompts=$_nsp"
   params="${params} --@num_target_prompts=-1"
   params="${params} --learn_attention=True"
   params="${params} --use_source_set=False"
   params="${params} --@source_prompts=$_src"
   params="${params} --attend_to_all=True"
   params="${params} --@learn_loaded_prompts=True#False"
   params="${params} --@use_private_prompts=$_upp"
   params="${params} --@load_private_prompts=$_lpp"
   params="${params} --@learn_attention=True"
   params="${params} --sel_positives=False"
   params="${params} --@learn_source_prompts=$_learn_sp"
   params="${params} --load_prompts=$_lp"
   params="${params} --@learn_loaded_prompts=True#!False"
   params="${params} --ignore_if_not_exist=False"
   params="${params} --rels=$_rels"
   params="${params} --@source_prompts_order=desc#rand"
   params="${params} --@num_random_masks=0"
   params="${params} --@compose_method=$_cmm"
   if [ -z "$_temp" ]; then
      params="${params} --@template=ptar"
   else
      params="${params} --@template=$_temp"
   fi
   params="${params} --attn_tuning=True#!False"
   params="${params} --attend_input=$_inp"
   if [ $_attn = "sub" ]; then
      params="${params} --attend_for=private#target"
   else
      params="${params} --attend_for=none#target"
   fi
   params="${params} --attend_source=True#!False"
   params="${params} --@add_target=$_addt"
   if [ $_addt = "True" ]; then
	params="${params} --@target_share=-1#0.5#0#none#1"
   fi
   params="${params} --@attend_target=False#True"
   params="${params} --@target_share_temperature=5."
   params="${params} --prompt_learning_rate=$_lr"
   if [ -n "$_slr" ]; then
      params="${params} --@source_prompt_learning_rate=$_slr"
   fi
   if [ -n "$_tlr" ]; then
      params="${params} --@target_prompt_learning_rate=$_tlr"
   fi
   if [ -n "$_plr" ]; then
      params="${params} --@private_prompt_learning_rate=$_plr"
   fi
   params="${params} --@attn_learning_rate=$_alr"
   params="${params} --@attn_method=$_attn"
   params="${params} --@temperature=$_tmpr"
   params="${params} --@anneal_dir=$_adir"
   params="${params} --normalize=True"
   params="${params} --anneal_min=0.0001"
   params="${params} --anneal_rate=none"
   params="${params} --@apply_softmax_to=after#!nothing"
   if [ -n "$_grm" ]; then
      params="${params} --@gen_route_methods=rb@direct@sigmoid"
   else
      params="${params} --@gen_route_methods=direct@"
   fi
   params="${params} --route_method=direct"
   params="${params} --use_saved_router=$_usr"
   params="${params} --init_from_words=False"
   params="${params} --prompts_to_save=none"
   params="${params} --save_router=$_sr"
   params="${params} --save_source_prompts=True"
   params="${params} --save_all_prompts=$_sp"
   params="${params} --router_prefix=$_rpx"
fi
echo "Learn: $_learn_sp, Load: $_lsp"
params=$(echo "$params" | sed "s/ --/\n --/g")
if [ -n "$_pv" ]; then
   echo "exp: ${exp} "
   echo $"Params: ${params} "
   echo "Run Params: ${run_params} "
   echo "Extra: ${extra_params}"
   exit
else
   echo "exp: ${exp} "
   echo "Run Params: ${run_params} "
   echo "Training ..."
   python3 run_seq2seq.py run ${run_params} -exp $exp ${params} ${extra_params} 
fi
if [ $? != 0 ] && [ "$onError" = "break" ];
then
   echo "exit 1"
   exit 1
fi
echo "EXIT 0"
done


if [ "$_exp" = "show" ]; then
   show_results --path=${log}/${exp}
fi

if [ "$_exp" != "self" ] && [ -z "$_file" ]; then
   cp train.sh ${log}/$exp
fi
case "$home" in 
   *-----TODO------*)
      # Do stuff
      mv /content/*time*.log ${log}/$exp
      tar -czvf /content/${exp}-$_exp.tar.gz ${log}/$exp
      cp /content/${exp}-$_exp.tar.gz ${home}/logs 
      ;;
esac

#mm  fa3 test2
