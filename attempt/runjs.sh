
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

if [ -z "$_bs" ]; then  _bs=8; fi

if [ -z "$_tn" ]; then  _tn=100#200#300; fi
if [ -z "$_vn" ]; then  _vn=20; fi
if [ -z "$_tsn" ]; then _tsn=100; fi
if [ -z "$_ep" ]; then  _ep=3#5#10; fi
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
for conf in ${PWD}/*.json; do
for method in $methods; do
echo "=============================== $method ========================="
params="${params} --post_config=!$conf"
var="method=$method"
var="${var}--data_path=atomic2020"
var="${var}--model_name_or_path=t5-base"
var="${var}--use_all_data=False"
var="${var}--test_ds_config=sel-test@" #@sel-test"
exp=$PWD
# operations
var="${var}--do_train=True"
var="${var}--do_test=True"
var="${var}--do_eval=True"
# Saving
var="${var}--save_total_limit=1"
var="${var}--save_checkpoint=False"
var="${var}--save_model=True"

# training 
var="${var}--per_device_train_batch_size=$_bs"
var="${var}--per_device_eval_batch_size=$_bs"
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
# xxxxxxxxxxxx
if [ "$method" = "px" ] || [ "$method" = "at" ]; then
	var="${var}--learning_rate=0.3"
	var="${var}--prefix_tuning=True"
	var="${var}--prefix_dim=100"
	var="${var}--opt_type=regular"
	var="${var}--use_optimizer=False"
        var="${var}--template=sup"
	params="${params} --prefix_dir=prefixes"
fi

if [ "$method" = "at" ]; then
        var="${var}--attn_prefix_tuning=True"
	var="${var}--prompt_embedding_path=xWant.pt@xNeed.pt@xIntent.pt"
	var="${var}--attn_method=sub"
fi
# pppppppppppp
# prompt tuning
if [ "$method" = "pt" ]; then
	var="${var}--prompt_tuning=True"
	var="${var}--use_optimizer=True"
	var="${var}--opt_type=regular"
	var="${var}--num_prompt_encoders=1"
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


