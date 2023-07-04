
#!/usr/bin/bash
shopt -s expand_aliases 
source ~/aa
####################
bash_params=""
global_run_params=""
extra_params=""
sep="#"
onError="break"
g=0
for i in $@
do
   case $i in
       _*) bash_params="${bash_params} $i"; g=bash;;
       g-*)
         i=${i#g}
         global_run_params="${global_run_params} $i";
         g=run
       ;;
      # Other Parameter 
       *) p=$i
       if [ "$g" = run ]
       then
          global_run_params="${global_run_params} $p"
          g=0
       elif [ "$g" = bash ]; then
          bash_params=${bash_params%=*}
          bash_params="${bash_params}=$i"
          g=arr
       elif [ "$g" = arr ]; then
          bash_params="${bash_params}${sep}$i"
          g=arr
       else
          others="$others $p"
       fi
       ;;
 esac
 if [ "$g" = bash ]; then
    bash_params="${bash_params}=1";
 fi
done

############################## Evaluate them
echo "Bash params: ${bash_params}"
echo "Global: $global_run_params"
eval "${bash_params}"

params=$bash_params
if [ -z "$_tasks" ]; then
  echo "No task provided, reading task from files ..."
  _tl=none
else
  _tl=$_tasks
  _tl=$(echo "$_tl" | sed "s/${sep}/ /g")
   echo "Tasks is $_tl"
fi
#if [ -n "$_rem" ]; then rm -rf ${log}/$_exp/*; fi
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
    if [ -z "$_def" ]; then
       params="${params} --prompt_learning_rate=$_lr"
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
    
