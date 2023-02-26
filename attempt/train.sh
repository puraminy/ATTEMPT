#!/bin/sh
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
         _attn=sub
         _adir=0#-1
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
      *"_atomic"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_atomic//")
         _cat=atomic
         _attn=rb
         _sph=3
         _task="xAttr@xIntent@xWant"
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
      *"_file"*) # debug is enabled
         _exp=file
         _file=True
         methods=files
         _train=False
         _pat=exp.json
         _lsp=True
         _lp=True
         bash_params=$(echo "$bash_params" | sed "s/_file//")
         cont=1
         ;;
      *"_unsup"*) # debug is enabled
         _template=unsup-p0-pt
         bash_params=$(echo "$bash_params" | sed "s/_unsup//")
         cont=1
         ;;
      *"_all"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_all//")
         _tn=-1
         _vn=20
         _tsn=-1 
         cont=1
         ;;
      *"_eval"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_eval//")
         _dotest=True
         _eval=True
         _train=False
         _lp=True
         cont=1
         ;;
      *"_g"*) # debug is enabled
         bash_params=$(echo "$bash_params" | sed "s/_g//")
         _cat=glue
         cont=1
         ;;
   esac
done
echo ${bash_params}
eval "${bash_params}"
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

if [ -z "$_cat" ]; then  _cat=general; fi
if [ -z "$_bs" ]; then  _bs=32; fi
if [ -z "$_learn_sp" ]; then  _learn_sp=True; fi
if [ -z "$_addt" ]; then  _addt=False; fi
if [ -z "$_attn" ]; then  _attn=rb; fi
if [ -z "$_adir" ]; then  _adir=-1; fi
if [ -z "$_template" ]; then 
   _template=sup-p0-pt
fi
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
   if [ -z "$_tn" ]; then  _tn=2; fi
   _vn=2
   _tsn=2 
   _ep=1
   _doeval=False
fi
if [ -z "$_train" ]; then  _train=True; fi
if [ -z "$_doeval" ]; then  _doeval=False; fi
if [ -z "$_dotest" ]; then  _dotest=True; fi
if [ -z "$_tn" ]; then  _tn=100; fi
if [ -z "$_vn" ]; then  _vn=50; fi
if [ -z "$_tsn" ]; then _tsn=100#200#500; fi
if [ -z "$_ep" ]; then  _ep=20; fi
if [ -z "$_err" ]; then  _err=break; fi
onError=$_err
if [ -z "$methods" ]; then
   methods=$(echo $others | xargs)
fi
params=""

if [ -z "$methods" ]; then
   methods="ptat"
fi
if [ "$_model" = "path" ]; then
   params="${params} --model_name_or_path=~${PWD}/trial=1"
fi

if [ "$_train" = "False" ]; then
   _tn=0
fi
if [ -z "$_pat" ]; then _pat=*.json; fi
for method in $methods; do
   echo "==================method: $method === epochs: $_ep ===== samples: $_train =========="
   # tttttt
   if [ -z "$_task" ]; then 
      _task="mnli@superglue-boolq@superglue-wic@superglue-wsc.fixed@superglue-cb"
   fi

   if [ -z "$_exp" ]; then 
      if [ -n "$_eval" ]; then 
         _exp="${_task}_eval"; 
      else
         _exp=$_task; 
      fi
   fi

   main_params=$params
   if [ "$method" = "files" ]; then
      if [ -n $_rem ]; then rm -rf ${log}/$_exp/*; fi
      for file in $PWD/$_pat; do
         echo "Config file=${file}"
         if [ -z $_file ]; then
            params="${main_params} --@task_name=$_task"
            params="${params} --test_ds_config=full-test@"
         fi
         params="${params} --per_device_train_batch_size=$_bs"
         params="${params} --per_device_eval_batch_size=$_bs"
         params="${params} --max_source_length=$_msl"
         params="${params} --max_target_length=$_mtl"
         params="${params} --load_source_prompts=$_lsp"
         params="${params} --load_prompts=$_lp"
         runat run ${run_params} -exp $_exp -cfg $file ${params} 
         if [ $? != 0 ] && [ "$onError" = "break" ]; then
            echo "exit 1"
            break
         fi
         #run_params=${run_params//"-rem"/}
      done
      break
   fi

   params="${params} --method=$method"

# data  ddddd
params="${params} --data_path=atomic2020"
params="${params} --use_all_data=False"
params="${params} --max_train_samples=$_tn"
params="${params} --max_val_samples=$_vn"
params="${params} --max_test_samples=$_tsn"
params="${params} --samples_per_head=$_sph"
#params="${params} --data_seed=123"
params="${params} --overwrite_cache=True"

# task
params="${params} --@task_name=$_task"
params="${params} --add_prefix=True"
params="${params} --ds_config=en@"
params="${params} --max_source_length=$_msl"
params="${params} --max_target_length=$_mtl"
params="${params} --test_ds_config=full-test@"

if [ "$_exp" = "self" ]; then
   # exp="${PWD#$log/}"
   exp=self
   _train=False
else
   exp=$method/$_cat/$_exp
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
# pppppppppppp
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

if [ "$method" = "pt" ]; then 
   params="${params} --prompt_learning_rate=0.1"
   params="${params} --num_prompt_tokens=24"
   params="${params} --prompt_encoder_type=mlp#emb#!lstm"
   params="${params} --prompt_sharing=!shared_tokens#shared_encoders"
   params="${params} --template=sup-p0-pt#unsup-p0-pt#!sup-p0-psh" 
   params="${params} --init_from_words=False"
   params="${params} --save_these_prompts=all"
fi
# aaaaaaaaaaaaaa
if [ "$method" = "ptat" ]; then
   #params="${params} --source_prompts=Is@it@grammatical@or@meaningful"
   params="${params} --prompt_encoder_type=mlp#!emb#!lstm"
   if [ -z $_lsp ]; then
      echo "Specify source prompting method"
      exit
   fi
   if [ "$_lsp" = "False" ]; then
      params="${params} --num_source_prompts=12"
      params="${params} --@num_target_prompts=8"
      params="${params} --load_source_prompts=False"
      params="${params} --learn_attention=True"
      params="${params} --learn_source_prompts=$_learn_sp"
   else
      params="${params} --source_prompts=mnli@qnli@stsb@qqp@rte@com1@com2@com3@com4"
      params="${params} --load_source_prompts=True"
      params="${params} --source_masking=True"
      params="${params} --num_target_prompts=8"
      params="${params} --@learn_attention=True"
      params="${params} --@^learn_source_prompts=$_learn_sp"
   fi
   params="${params} --compose_method=cat#wavg"
   params="${params} --template=$_template#!sup-p0-pt#!unsup-p0-psh#!sup-p0-psh" 
   params="${params} --attn_tuning=True"
   params="${params} --attend_input=$_ai"
   params="${params} --attend_source=True#!False"
   params="${params} --softmax_sel=True"
   params="${params} --add_target=$_addt"
   if [ "$_addt" = "True" ]; then
      params="${params} --attend_target=False"
      params="${params} --@target_share=-1"
      params="${params} --target_share_temperature=.1"
   else
      params="${params} --attend_target=True"
   fi
   params="${params} --prompt_learning_rate=0.1"
   params="${params} --attn_learning_rate=0.01"
   params="${params} --attn_method=$_attn"
   params="${params} --@anneal_dir=$_adir"
   params="${params} --sig_coef=1"
   params="${params} --temperature=10"
   params="${params} --anneal_min=10e-10"
   params="${params} --anneal_rate=none"
   params="${params} --num_prompt_tokens=3"
   params="${params} --gen_route_methods=rb@" #@sigmoid@sign"
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


