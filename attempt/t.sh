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
home=$HOME
case "$HOME" in 
  *ahmad*)
    # Do stuff
    model=t5-base
    ;;
  *root*)
    # Colab 
    home=/content/drive/MyDrive
    ;;
esac
case $run_params in
  *"-d"*) # debug is enabled
      _test=True
    ;;
esac

alias show_results="python3 ${home}/mt5-comet/comet/train/show.py full"
alias runat="python3 ${home}/ATTEMPT/attempt/run_seq2seq.py"
cont=1
while [ "$cont" -eq 1 ]; do
cont=0
case $bash_params in
  *"_learn"*) # debug is enabled
      _learn_sp=True
      bash_params=$(echo "$bash_params" | sed "s/_learn//")
      cont=1
    ;;
  *"_test"*) # debug is enabled
      _test=True
      bash_params=$(echo "$bash_params" | sed "s/_test//")
      _lsp=False
      cont=1
    ;;
  *"_self"*) # debug is enabled
      _exp=self
      bash_params=$(echo "$bash_params" | sed "s/_self//")
      cont=1
    ;;
esac
done
echo ${bash_params}
eval "${bash_params}"

# wrap experiments
folder=${PWD##*/}          
log=${home}/logs   
echo "log: ${log}"

if [ -z "$_bs" ]; then  _bs=32; fi
if [ -z "$_learn_sp" ]; then  _learn_sp=False; fi
if [ -z "$_sp" ]; then  _sp=none; fi
if [ -z "$_lsp" ]; then  _lsp=True; fi
if [ -z "$_lp" ]; then  _lp=False; fi

# eeeee
if [ -n "$_test" ]; then
  _rem=True
  if [ -z "$_tn" ]; then  _tn=2; fi
  _vn=2
  _tsn=2 
  _ep=1
  _eval=False
fi
if [ -n "$_all" ]; then
  _tn=-1
  _vn=20
  _tsn=-1 
fi
if [ -z "$_train" ]; then  _train=True; fi
if [ -z "$_eval" ]; then  _eval=False; fi
if [ -z "$_tn" ]; then  _tn=100; fi
if [ -z "$_vn" ]; then  _vn=50; fi
if [ -z "$_tsn" ]; then _tsn=100; fi
if [ -z "$_ep" ]; then  _ep=10; fi
onError=break
methods=$(echo $others | xargs)
params=""

if [ -z "$methods" ]; then
  methods="ptat"
fi
if [ "$_model" = "path" ]; then
   params="${params} --model_name_or_path=~${PWD}/trial=1"
fi

if [ -z "$_exp" ]; then _exp=noname; fi
if [ -z "$_pat" ]; then _pat=*.json; fi
for method in $methods; do
echo "==================method: $method === epochs: $_ep ===== samples: $_train =========="
# tttttt
if [ -z "$_task" ]; then 
_task="xAttr@xIntent@xWant"
#_task="cola"
fi

main_params=$params
if [ "$method" = "files" ]; then
   if [ -n $_rem ]; then rm -rf ${log}/$_exp/*; fi
   for file in $PWD/$_pat; do
	echo "Config file=${file}"
	params="${main_params} --@task_name=$_task"
	params="${params} --test_ds_config=full-test@"
	params="${params} --per_device_train_batch_size=$_bs"
	params="${params} --per_device_eval_batch_size=$_bs"
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
params="${params} --@max_train_samples=$_tn"
params="${params} --max_val_samples=$_vn"
params="${params} --max_test_samples=$_tsn"
#params="${params} --data_seed=123"
params="${params} --overwrite_cache=True"

# task
params="${params} --@task_name=$_task"
params="${params} --add_prefix=True"
params="${params} --ds_config=en@"
params="${params} --test_ds_config=full-test@"

exp=$_exp
if [ "$_exp" = "self" ]; then
  exp="${PWD#$log/}"
  _train=False
fi

params="${params} --do_train=$_train"
# operations
params="${params} --do_test=True"
params="${params} --do_eval=$_eval"
# Saving
params="${params} --report_to=wandb@"
params="${params} --save_total_limit=1"
params="${params} --save_checkpoint=False"
params="${params} --save_model=False"
params="${params} --load_best_model_at_end=True" # We handle it by loading best prompts

# training 
params="${params} --per_device_train_batch_size=$_bs"
params="${params} --per_device_eval_batch_size=$_bs"
params="${params} --trainer_shuffle=True"
params="${params} --skip_specials=True"
params="${params} --num_train_epochs=$_ep"
params="${params} --adjust_epochs=True"

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
if [ -z "$_ppx" ]; then  _ppx=$_exp; fi

# prompt tuning common settings
if [ "$method" = "pt" ] || [ "$method" = "ptat" ]; then
	params="${params} --prompt_tuning=True"
	params="${params} --use_optimizer=True"
	params="${params} --opt_type=regular"
	params="${params} --prompt_encoders_dir=prompts"
	params="${params} --load_prompts=$_lp"
	params="${params} --prompts_prefix=$_ppx"
fi

if [ "$method" = "pt" ]; then 
	params="${params} --prompt_learning_rate=0.01"
	params="${params} --num_prompt_tokens=5#3#1"
	params="${params} --prompt_encoder_type=mlp#emb#!lstm"
	params="${params} --prompt_sharing=!shared_prompts#shared_encoders"
        params="${params} --template=unsup-p0-psh#unsup-p0-pt#!sup-p0-psh" 
	params="${params} --init_from_words=False"
	params="${params} --save_prompts=all"
fi
# aaaaaaaaaaaaaa
if [ "$method" = "ptat" ]; then
        #params="${params} --source_prompts=Is@it@grammatical@or@meaningful"
	params="${params} --@prompt_encoder_type=mlp#!emb#!lstm"
	#params="${params} --source_prompts=adj@always@they@seen@event@before@want@after"
	params="${params} --@compose_method=cat#wavg"
	params="${params} --@num_prompt_tokens=5"
	params="${params} --@select_method=prior"
        params="${params} --@template=unsup-p0-pt#!sup-p0-pt#!unsup-p0-psh#!sup-p0-psh" 
	params="${params} --load_source_prompts=$_lsp"
	params="${params} --learn_source_prompts=$_learn_sp"
	params="${params} --attn_tuning=True"
	params="${params} --@attend_input=False#True"
	params="${params} --attend_source=True#!False"
	params="${params} --@attend_target=False#True"
	params="${params} --@add_target=False#True"
	params="${params} --@prompt_learning_rate=0.1#0.1"
	params="${params} --@^attn_learning_rate=0.01#0.001"
	params="${params} --@attn_method=rb#!sub#!linear"
	params="${params} --@anneal_dir=-0.001#-1"
	params="${params} --@^router_temperature=4."
	params="${params} --anneal_min=0.001"
	params="${params} --anneal_rate=none"
	params="${params} --@num_source_prompts=8#8"
	params="${params} --@^num_target_prompts=4#8"
	params="${params} --@gen_route_methods=rb@sigmoid@sign"
	params="${params} --save_prompts=$_sp"
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

if [ "$_exp" != "self" ]; then
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


