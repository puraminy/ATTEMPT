
#!/bin/sh

params=""
g2=""
for i in $@
do
   case $i in
       # -- option
       --*) params="${params} $i"; g=1;;
       
       -m) echo "------"; g=3;;
       # - option
       -*) g2="${g2} $i"; g=2;;
       
       # Parameter 
       *) p=$i
          if [ "$g" = 1 ]
          then
            params="${params} $p"
            g=0
          elif [ "$g" = 2 ]
          then
            g2="${g2} $p"
            g=0
          elif [ "$g" = 3 ]
          then
            m=$p 
            g=0
          else
            others="$others $p"
          fi
      ;;
   esac
done
echo "Others: ${others}"
model=t5-base
case "$HOME" in 
  *ahmad*)
    # Do stuff
    model=t5-base
    ;;
esac

home=$HOME
config=$HOME/ATTEMPT/attempt/configs/baselines/base.json 
alias runat="python3 ${home}/ATTEMPT/attempt/run_seq2seq.py"
# wrap experiments
folder=${PWD##*/}          
if [ -z $m ]; then
   m=11
fi
echo "m: ${m}"
if [ "$m" = "0" ]; then
  echo "testing train"
elif [ "$m" = "test" ]; then
  echo "testing train and test"
  train_num=2
  val_num=1
  test_num=1 
  epochs=1
fi
log=${home}/logs   
echo "log: ${log}"

# tehran data 
#

if [ "$m" = "test" ]; then
   echo "testing"
else
  train_num=-1
  val_num=10
  test_num=-1
  epochs=8
fi
onError=break
params="${params} --model_name_or_path=!${PWD}"
echo $params

methods=$(echo $others | xargs)
if [ -z "$methods" ]; then
  methods="ft pt px"
fi
for method in $methods; do
echo "=============================== $method ========================="
var="method=$method"
var="${var}--data_path=atomic2020"
var="${var}--use_all_data=False"
var="${var}--max_train_samples=$train_num"
var="${var}--max_val_samples=$val_num"
var="${var}--max_test_samples=$test_num"
var="${var}--data_seed=42"
var="${var}--overwrite_cache=True"

# task
task="cola@"
var="${var}--task_name=$task"
var="${var}--ds_config=en@"

var="${var}--test_dataset_name=$task"
var="${var}--test_ds_config=full-test@" #@sel-test"

exp=$task-$m

# operations
var="${var}--do_train=False"
var="${var}--do_test=True"
var="${var}--do_eval=True"

# Saving
var="${var}--save_total_limit=1"
var="${var}--save_checkpoint=True"

# training 
var="${var}--num_train_epochs=$epochs"
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
fi

# prefix tuning
if [ "$method" = "px" ]; then
	var="${var}--learning_rate=0.3"
	var="${var}--prefix_tuning=True"
	var="${var}--prefix_dim=100"
        var="${var}--per_device_train_batch_size=32"
	var="${var}--use_optimizer=False"
        var="${var}--template=sup"
fi

# prompt tuning
if [ "$method" = "pt" ]; then
	var="${var}--prompt_tuning=True"
	var="${var}--use_optimizer=True"
	var="${var}--opt_type=regular"
	var="${var}--prompt_learning_rate=0.1#0.01"
	var="${var}--num_prompt_encoders=1"
        var="${var}--per_device_train_batch_size=32"
	var="${var}--num_prompt_tokens=8"
	var="${var}--prompt_encoder_type=lstm"
        var="${var}--template=sup-pt-t"
fi
runat run $g2 -exp $exp -cfg $config -var ${var} ${params} 
if [ $? != 0 ] && [ "$onError" = "break" ];
then
    echo "exit 1"
    break
fi
echo "EXIT 0"
done

alias show_results="python3 /home/pouramini/mt5-comet/comet/train/show.py full "
if [ "$m" = "show" ]; then
   show_results --path=${log}/${exp}
fi

case "$home" in 
  *content*)
    # Do stuff
	mv /content/*time*.log ${log}/$exp
	tar -czvf /content/${exp}-$m.tar.gz ${log}/$exp
	cp /content/${exp}-$m.tar.gz ${home}/logs 
    ;;
esac


