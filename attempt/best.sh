
#!/bin/sh

g1=""
g2=""
for i in $@
do
   case $i in
       # -- option
       --*) g1="${g1} $i"; g=1;;
       
       -m) echo "------"; g=3;;
       # - option
       -*) g2="${g2} $i"; g=2;;
       
       # Parameter 
       *) p=$i
          if [ "$g" = 1 ]
          then
            g1="${g1} $p"
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
config=$(echo $others | xargs)
model=t5-base
case "$HOME" in 
  *ahmad*)
    # Do stuff
    model=t5-base
    ;;
esac
if [ -z $config ]; then
   config=configs/baselines/base.json 
fi 
echo "Config: ${config}"
home=$HOME
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
  train_num=100#200
  val_num=10
  test_num=100
  epochs=3
fi

var="data_path=logs/xattr-1" #mt5-comet/comet/data/atomic2020"
var="data_path=mt5-comet/comet/data/atomic2020"

var="${var}--use_all_data=False"
var="${var}--max_train_samples=$train_num"
var="${var}--max_val_samples=$val_num"
var="${var}--max_test_samples=$test_num"
var="${var}--data_seed=123"

# task
task="xIntent@#xAttr@"
var="${var}--task_name=$task"
var="${var}--ds_config=en@en"
var="${var}--template=task-mid" #task-mid-nat#task-mid-nat2" 

exp=$task-$m

# operations
var="${var}--do_train=True"
var="${var}--do_test=True"
var="${var}--do_eval=False"

# training 
var="${var}--learning_rate=0.0003"
var="${var}--!use_optimizer=False#True"
var="${var}--num_train_epochs=$epochs"
var="${var}--per_device_train_batch_size=8"
var="${var}--per_device_eval_batch_size=8"
var="${var}--trainer_shuffle=True#False"
var="${var}--opt_type=sep#regular"

# Saving
var="${var}--save_total_limit=1"

# prefix tuning
var="${var}--prefix_tuning=False"
var="${var}--prefix_dim=100"

# prompt tuning
var="${var}--prompt_tuning=True"
var="${var}--prompt_learning_rate=0.1#0.2"
var="${var}--num_prompt_encoders=1"
var="${var}--num_prompt_tokens=8"
var="${var}--prompt_encoder_type=lstm"


runat run $g2 -exp $exp -cfg $config -var ${var} 

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


