
#!/bin/sh

params=""
run_params=""
bash_params=""
for i in $@
do
   case $i in
       --*) params="${params} $i"; g=1;;
       
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
echo "Others: ${others}"
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
eval ${bash_params}
alias runat="python3 ${home}/ATTEMPT/attempt/run_seq2seq.py"
# wrap experiments
folder=${PWD##*/}          
log=${home}/logs   
echo "log: ${log}"

if [ -z "$_tn" ]; then  _tn=200; fi
if [ -z "$_vn" ]; then  _vn=20; fi
if [ -z "$_tsn" ]; then _tsn=100; fi
if [ -z "$_ep" ]; then  _ep=3; fi
if [ -n "$_test" ]; then
  _tn=10
  _vn=2
  _tsn=2 
  _ep=1
fi
if [ -n "$_all" ]; then
  _tn=-1
  _vn=20
  _tsn=-1 
fi
onError=break
methods=$(echo $others | xargs)
if [ -z "$methods" ]; then
  methods="ft pt px"
fi
if [ "$_model" = "path" ]; then
   params="${params} --model_name_or_path=!${PWD}/trial=1"
fi

for method in $methods; do
echo "=============================== $method ========================="
var="method=$method"
var="${var}--data_path=atomic2020"
var="${var}--use_all_data=False"
var="${var}--max_train_samples=$_tn"
var="${var}--max_val_samples=$_vn"
var="${var}--max_test_samples=$_tsn"
var="${var}--data_seed=123"
var="${var}--overwrite_cache=True"

# task
task="xIntent@"
var="${var}--task_name=$task"
var="${var}--ds_config=en@"

var="${var}--test_dataset_name=$task"
var="${var}--test_ds_config=full-test@" #@sel-test"

exp=$task-$_exp
if [ "$_exp" = "self" ]; then
  exp="${PWD#$log/}"
  echo "cur folder: ${exp}"
  var="${var}--do_train=False"
else
  var="${var}--do_train=True"
fi
# operations
var="${var}--do_test=True"
var="${var}--do_eval=True"
# Saving
var="${var}--save_total_limit=1"
var="${var}--save_checkpoint=True"
var="${var}--save_model=True"

# training 
var="${var}--per_device_train_batch_size=8"
var="${var}--per_device_eval_batch_size=8"
var="${var}--trainer_shuffle=True"
var="${var}--skip_specials=True"
var="${var}--load_best_model_at_end=True"


if [ "$method" = "ft" ]; then
	var="${var}--learning_rate=0.0003"
	var="${var}--opt_type=regular"
        var="${var}--per_device_train_batch_size=16"
        var="${var}--template=sup"
	var="${var}--num_train_epochs=$_ep"
fi

# prefix tuning
# xxxxxxxxxxxx
if [ "$method" = "px" ]; then
	var="${var}--learning_rate=0.3"
	var="${var}--prefix_tuning=True"
	var="${var}--prefix_dim=100"
        var="${var}--per_device_train_batch_size=8"
	var="${var}--opt_type=regular"
	var="${var}--use_optimizer=False"
        var="${var}--template=sup"
	var="${var}--num_train_epochs=$_ep"
	var="${var}--config=base#attempt"
	params="${params} --prefix_dir=prefixes"
fi

# pppppppppppp
# prompt tuning
if [ "$method" = "pt" ]; then
	var="${var}--prompt_tuning=True"
	var="${var}--use_optimizer=True"
	var="${var}--opt_type=regular"
	var="${var}--prompt_learning_rate=0.1"
	var="${var}--num_prompt_encoders=1"
        var="${var}--per_device_train_batch_size=8"
	var="${var}--num_prompt_tokens=8"
	var="${var}--prompt_encoder_type=mlp"
        var="${var}--template=unsup-pt-t"
	var="${var}--num_train_epochs=$_ep"
	params="${params} --prompt_encoders_dir=prompts"
fi
echo "other params: ${params}"
runat run ${run_params} -exp $exp -var ${var} ${params} 
if [ $? != 0 ] && [ "$onError" = "break" ];
then
    echo "exit 1"
    break
fi
echo "EXIT 0"
done

alias show_results="python3 /home/pouramini/mt5-comet/comet/train/show.py full "
if [ "$_exp" = "show" ]; then
   show_results --path=${log}/${exp}
fi

if [ "$_exp" != "self" ]; then
	cp train.sh ${log}/$exp
fi
case "$home" in 
  *content*)
    # Do stuff
	mv /content/*time*.log ${log}/$exp
	tar -czvf /content/${exp}-$_exp.tar.gz ${log}/$exp
	cp /content/${exp}-$_exp.tar.gz ${home}/logs 
    ;;
esac


