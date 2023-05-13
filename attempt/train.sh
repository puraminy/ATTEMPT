#!/bin/bash
extra_params=""
run_params=""
bash_params=""
for i in $@
do
   case $i in
      --*) extra_params="${extra_params} $i"; g=1;;
      _*) bash_params="${bash_params} $i"; g=3;;
      -*) run_params="${run_params} $i"; g=2;;

       # Parameter 
       *) p=$i
       if [ "$g" = 2 ]
       then
          run_params="${run_params} $p"
          g=0
       else
          others="$others $p"
       fi
       ;;
 esac
done
model=t5-base
tname=t5-base
home=$HOME
case "$HOME" in 
   *root*)
      # Colab 
      home=/content/drive/MyDrive
      ;;
esac
alias show_results="python3 ${home}/ATTEMPT/attempt/show.py full"
alias runat="python3 ${home}/ATTEMPT/attempt/run_seq2seq.py"
cont=1
while [ "$cont" -eq 1 ]; do
   cont=0
   case $bash_params in
      *"_mt5"*) # debug is enabled
         model=mt5-base
         tname=mt5-base
         bash_params=$(echo "$bash_params" | sed "s/_mt5//")
         cont=1
         ;;
      *"_pars"*) # debug is enabled
         model=parsT5-base
         tname=parsT5-base
         bash_params=$(echo "$bash_params" | sed "s/_pars//")
         cont=1
         ;;
      *"_v1"*) # debug is enabled
         model=google/t5-v1_1-base
         tname=t5-base
         bash_params=$(echo "$bash_params" | sed "s/_v1//")
         cont=1
         ;;
      *"_large"*) # debug is enabled
         model=t5-large
         tname=t5-large
         bash_params=$(echo "$bash_params" | sed "s/_large//")
         cont=1
         ;;
      *"_load"*) # load source prompts
         _lsp=True
         bash_params=$(echo "$bash_params" | sed "s/_load//")
         _ai=False
         _addt=False
         _learn_sp=True
         cont=1
         ;;
      *"_learn"*) # learn source prompts
         _learn_sp=True
         bash_params=$(echo "$bash_params" | sed "s/_learn//")
         _ai=False
         _addt=False
         cont=1
         ;;
      *"_max"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_max//")
         _msl=450
         _mtl=25
         cont=1
         ;;
      *"_glue"*) # debug is enabled
         _learn_sp=True
         _attn="rb"
         _cmm="wavg#cat"
         _glue=True
         _adir=-1
         bash_params=$(echo "$bash_params" | sed "s/_glue//")
         _ai=True
         _addt=True
         _msl=450
         _mtl=25
         #_tsn=-1
         _lsp=False
         cont=1
         ;;
      *"_test"*) # debug is enabled
         _test=True
         bash_params=$(echo "$bash_params" | sed "s/_test//")
         _lsp=False
         cont=1
         ;;
      *"_seed"*) # debug is enabled
         _seed=True
         bash_params=$(echo "$bash_params" | sed "s/_seed//")
         cont=1
         ;;
      *"_adapter"*) # debug is enabled
         _adapter=True
         bash_params=$(echo "$bash_params" | sed "s/_adapter//")
         cont=1
         ;;
      *"_arels"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_arels//")
         _cat=atomic
         _cmm="cat"
         _atomic=True
         _rels="xWant@xAttr"
         _attn="rb"
         _sph=3
         _task="atomic-rels" #xAttr@xIntent@xWant@xReact" 
         #_temp="sup-p0-px0-px-pt-rel"
         _temp="sup-px-pt-rel"
         #@xEffect@oEffect@oReact@xNeed"
         _addt=False
         _lsp=False
         cont=1
         ;;

      *"_sgtasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_sgtasks//")
         _task="superglue-wsc.fixed@superglue-cb@superglue-boolq@superglue-rte@superglue-copa@superglue-wic"
         cont=1
         ;;
      *"_sep-sgtasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_sep-sgtasks//")
         _task="superglue-wsc.fixed#superglue-cb#superglue-boolq#superglue-rte#superglue-copa#superglue-wic"
         cont=1
         ;;
      *"_atasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_atasks//")
         _task="xAttr@xIntent@Causes@Desires@xReason@xWant@oReact@xNeed"
         cont=1
         ;;
      *"_sep-atasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_sep-atasks//")
         _task="xAttr#xIntent#Causes#Desires#xReason#xWant#oReact#xNeed"
         cont=1
         ;;
      *"_gtasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_gtasks//")
         _task="mrpc@sst2@qnli@qqp@rte@stsb@cola@mnli"
         cont=1
         ;;
      *"_minus-gtasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_minus-gtasks//")
         _task="mrpc@sst2@qnli@qqp@rte@stsb@cola"
         cont=1
         ;;
      *"_sep-gtasks"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_sep-gtasks//")
         _task="mrpc#sst2#qnli#qqp#rte#stsb#cola#mnli"
         cont=1
         ;;
      *"_atomic"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_atomic//")
         _cat=atomic
         _cmm="wavg#cat"
         _atomic=True
         _attn="rb"
         _sph=3
         _task="xAttr@xIntent@xWant" #xAttr@xIntent@xWant@xReact" 
         #@xEffect@oEffect@oReact@xNeed"
         _addt=False
         _lsp=False
         cont=1
         ;;
      *"_self"*) # debug is enabled
         _exp=self
         _self=True
         bash_params=$(echo "$bash_params" | sed "s/_self//")
         cont=1
         ;;
      *"_files"*) # debug is enabled
         _met=files
         bash_params=$(echo "$bash_params" | sed "s/_files//")
         cont=1
         ;;
      *"_unsup"*) 
         _temp=unsup-p0-px0-px-pt
         bash_params=$(echo "$bash_params" | sed "s/_unsup//")
         cont=1
         ;;
      *"_pt"*) 
         _met="pt"
         bash_params=$(echo "$bash_params" | sed "s/_pt//")
         cont=1
         ;;
      *"_all"*) 
         bash_params=$(echo "$bash_params" | sed "s/_all//")
         _tn=-1
         _vn=20
         _tsn=-1 
         cont=1
         ;;
      *"_eval"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_eval//")
         _dotest=True
         _doeval=True
         _train=False
         _lp=True
         cont=1
         ;;
   esac
done
echo ${bash_params}
eval "${bash_params}"
echo "Tasks: $others"
case $run_params in
   *"-d"*) # debug is enabled
      _test=True
      ;;
   *"-nd"*) # debug is enabled
      run_params=$(echo "$run_params" | sed "s/-nd/-d/")
      ;;
esac
case "$HOME" in 
   *ahmad*)
      # Do stuff
      model=t5-base
      tname=t5-base
      ;;
esac
# wrap experiments
folder=${PWD##*/}          
log=${home}/logs   
echo "log: ${log}"

if [ -z "$_bs" ]; then  _bs=32; fi
if [ -z "$_learn_sp" ]; then  _learn_sp=True; fi
if [ -z "$_addt" ]; then  _addt=False; fi
if [ -z "$_attn" ]; then  _attn=rb; fi
if [ -z "$_cmm" ]; then  _cmm="wavg#cat"; fi
if [ -z "$_at" ]; then  _at=True; fi
if [ -z "$_adir" ]; then  _adir=-1; fi
if [ -z "$_sp" ]; then  _sp=none; fi
if [ -z "$_ai" ]; then  _ai=False; fi
if [ -z "$_lp" ]; then  _lp=False; fi
if [ -z "$_pdir" ]; then  _pdir=prompts; fi
if [ -z "$_msl" ]; then  _msl=200; fi
if [ -z "$_mtl" ]; then  _mtl=120; fi
if [ -z "$_sph" ]; then  _sph=1; fi

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
if [ -z "$_tn" ]; then  _tn=200; fi
if [ -z "$_vn" ]; then  _vn=50; fi
if [ -z "$_tsn" ]; then _tsn=100; fi
if [ -z "$_ep" ]; then  _ep=20; fi
if [ -z "$_err" ]; then  _err=break; fi
onError=$_err
params=""

if [ -z "$_met" ]; then
   _met="ptat"
fi
if [ "$_model" = "path" ]; then
   params="${params} --model_name_or_path=~${PWD}/trial=1"
fi

if [ "$_train" = "False" ]; then
   _tn=0
fi
if [ -z "$_pat" ]; then _pat=exp.json; fi
for method in $_met; do
   echo "==================method: $method === epochs: $_ep ===== samples: $_train =========="
   if [ -z "$_exp" ]; then 
      if [ -n "$_eval" ]; then 
         _exp="${_task}_eval"; 
      else
         _exp=$_task; 
      fi
   fi

   main_params=$params
   if [ "$method" = "files" ]; then
      if [ "$others" = "" ]; then
         echo "No task provided, reading task from files ..."
         _tl=none
      else
         _tl=$others
         echo "Tasks is $_tl"
         #params="${params} --test_ds_config=full-test@"
         #params="${params} --max_source_length=$_msl"
         #params="${params} --max_target_length=$_mtl"
      fi
      #if [ -n $_rem ]; then rm -rf ${log}/$_exp/*; fi
      for task in $_tl; do
         find ${PWD} -maxdepth 2 -type f -iname "$_pat*.json" | while read file
         do
            echo "Config file=${file} for task $task"
            if [ $task = "none" ]; then
               echo "task is none"
            else
               params="${main_params} --@task_name=$task"
            fi
            #params="${params} --rels=xAttr@xWant"
            if [ -z $_seed ]; then
               echo "default seed"
            else
               params="${params} --@data_seed=123"
            fi
            if [ -z $_def ]; then
               params="${params} --source_prompt_learning_rate=0.001"
               params="${params} --target_prompt_learning_rate=0.001"
               params="${params} --per_device_train_batch_size=$_bs"
               params="${params} --per_device_eval_batch_size=$_bs"
               # params="${params} --@max_train_samples=$_tn"
               # params="${params} --max_val_samples=$_vn"
               params="${params} --max_test_samples=$_tsn"
               params="${params} --@num_train_epochs=$_ep"
            fi 
            # operations
            params="${params} --do_train=$_train"
            params="${params} --do_test=$_dotest"
            params="${params} --do_eval=$_doeval"
            if [ -z $_exp ]; then
               exp=$task
            else
               exp=$_exp
            fi
            if [ -n "$_cat" ]; then
               exp=$_cat/$exp
            else
               exp=$_pat/$exp
            fi
            runat run ${run_params} -exp $exp -cfg $file ${params} ${extra_params} 
            if [ $? != 0 ] && [ "$onError" = "break" ]; then
               echo "exit 1"
               break
            fi
            #run_params=${run_params//"-rem"/}
         done
      done
      break
   fi

   params="${params} --method=$method"

# data  ddddd
params="${params} --data_path=atomic2020"
params="${params} --use_all_data=False"
params="${params} --@max_train_samples=$_tn"
params="${params} --max_val_samples=$_vn"
params="${params} --max_test_samples=$_tsn"
params="${params} --samples_per_head=$_sph"
#params="${params} --data_seed=123"
params="${params} --overwrite_cache=True"
if [ -z $_seed ]; then
   params="${params} --@data_seed=123"
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
params="${params} --tokenizer_name=$tname"
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

if [ "$method" = "ft" ]; then
   params="${params} --learning_rate=0.0003"
   params="${params} --opt_type=regular"
   params="${params} --per_device_train_batch_size=16"
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
if [ -z "$_ppx" ]; then  _ppx=none; fi

# prompt tuning common settings
if [ "$method" = "pt" ] || [ "$method" = "ptat" ]; then
   params="${params} --prompt_tuning=True"
   params="${params} --use_optimizer=True"
   params="${params} --opt_type=regular"
   params="${params} --prompt_encoders_dir=$_pdir"
   params="${params} --load_prompts=$_lp"
   params="${params} --prompts_prefix=$_ppx"
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
   params="${params} --source_prompt_learning_rate=0.01"
   params="${params} --target_prompt_learning_rate=0.01"
   params="${params} --@num_prompt_tokens=40"
   params="${params} --@prompt_encoder_type=mlp#!lstm"
   if [ -z $_temp ]; then
      params="${params} --@template=0-pcom-pt#0-pt-pt"
   else
      params="${params} --template=$_temp"
   fi
   params="${params} --@num_prompts=2"
   params="${params} --@task_comb=none"
   params="${params} --@prompt_sharing=shared_encoders"
   params="${params} --init_from_words=False"
   params="${params} --load_source_prompts=False"
   params="${params} --load_prompts=True"
   params="${params} --ignore_if_not_exist=True"
   params="${params} --@learn_loaded_prompts=False"
   params="${params} --save_these_prompts=all"
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
   params="${params} --@num_prompt_tokens=40"
   params="${params} --@num_source_prompts=0"
   params="${params} --@num_target_prompts=-1"
   params="${params} --learn_attention=True"
   params="${params} --use_prompt_set=False"
   params="${params} --@source_prompts=$_src"
   params="${params} --@learn_loaded_prompts=True#False"
   params="${params} --@use_private_prompts=False#True"
   params="${params} --@learn_attention=True"
   params="${params} --sel_positves=False"
   params="${params} --@learn_source_prompts=$_learn_sp"
   params="${params} --load_prompts=True"
   params="${params} --@learn_loaded_prompts=True#False"
   params="${params} --ignore_if_not_exist=True"
   params="${params} --rels=$_rels"
   params="${params} --@source_prompts_order=rand#desc"
   params="${params} --@num_random_masks=0"
   params="${params} --@compose_method=$_cmm"
   params="${params} --@template=$_temp"
   params="${params} --attn_tuning=True#!False"
   params="${params} --attend_input=False#True"
   params="${params} --attend_for=none#inp_target"
   params="${params} --attend_source=True#!False"
   params="${params} --@apply_softmax_to=after"
   params="${params} --@add_target=False"
   params="${params} --@target_share=none#0.5#0#-1#1"
   params="${params} --@attend_target=False#True"
   params="${params} --@target_share_temperature=1."
   params="${params} --source_prompt_learning_rate=0.01"
   params="${params} --target_prompt_learning_rate=0.01"
   params="${params} --attn_learning_rate=0.0001"
   params="${params} --@attn_method=$_attn"
   params="${params} --@anneal_dir=-1"
   params="${params} --temperature=1."
   params="${params} --normalize=True"
   params="${params} --anneal_min=10e-10"
   params="${params} --anneal_rate=none"
   params="${params} --@gen_route_methods=rb@"
   params="${params} --route_method=rb"
   params="${params} --init_from_words=False"
   params="${params} --save_these_prompts=$_sp"
   params="${params} --save_source_prompts=True"
fi
echo "Learn: $_learn_sp, Load: $_lsp"
runat run ${run_params} -exp $exp ${params} ${extra_params} 
if [ $? != 0 ] && [ "$onError" = "break" ];
then
   echo "exit 1"
   break
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


