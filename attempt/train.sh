
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

train_num=-1
val_num=-1
test_num=-1
epochs=3

if [ -z $m ]; then
   m=11
fi
echo "m: ${m}"
if [ "$m" -eq "0" ]; then
  echo "testing train"
elif [ "$m" -eq "1" ]; then
  echo "testing train and test"
  train_num=2
  val_num=1
  test_num=1 
  epochs=1
fi
log=${home}/logs   
echo "log: ${log}"

# local data 
var="data_path=logs/xattr-1#mt5-comet/comet/data/atomic2020"

var="${var}--use_all_data=False"
var="${var}--max_train_samples=$train_num"
var="${var}--max_val_samples=$val_num"
var="${var}--max_test_samples=$test_num"

# task
task="xAttr@"
var="${var}--task_name=$task"
var="${var}--eval_dataset_name=$task" 
var="${var}--test_dataset_name=$task" 

exp=att-$task-$m
exp=xattr-2

# operations
var="${var}--do_train=True"
var="${var}--do_test=True"
var="${var}--do_eval=False"

# training 
var="${var}--learning_rate=0.3#0.5#0.01#0.1"
var="${var}--use_optimizer=False"
var="${var}--num_train_epochs=$epochs"
var="${var}--per_device_train_batch_size=8"
var="${var}--per_device_eval_batch_size=8"

# Saving
var="${var}--save_total_limit=0"

# prefix tuning
var="${var}--prefix_tuning=False"
var="${var}--prefix_dim=100"

# prompt tuning
var="${var}--prompt_tuning=True"
var="${var}--prompt_learning_rate=0.5"
var="${var}--num_prompt_encoders=1"
var="${var}--num_prompt_tokens=8"
var="${var}--prompt_encoder_type=lstm"


runat run $g2 -exp $exp -cfg $config -var ${var} 

alias show_results="python3 /home/pouramini/mt5-comet/comet/train/show.py full "
show_results --path=${log}/${exp}

case "$home" in 
  *content*)
    # Do stuff
	mv /content/*time*.log ${log}/$exp
	tar -czvf /content/${exp}-$m.tar.gz ${log}/$exp
	cp /content/${exp}-$m.tar.gz ${home}/logs 
    ;;
esac


