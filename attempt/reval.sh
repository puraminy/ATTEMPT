#!/usr/bin/bash
shopt -s expand_aliases 
source ~/aa
extra_params=""
vars=""
flags=""
others=""
onError="continue"
main_vars=""
g=0
for i in $@
do
   case $i in
      --*) 
            q=${i#"--"}
            extra_params="${extra_params} --@${q}"; 
            p=${i%=*}
            main_vars="${main_vars}@${q}"
            g=extra
            mvar=True
       ;;
       -*) extra_params="${extra_params} $i"; g=run;;
       *=*) vars="${vars} $i"; g=vars;;
       _*) flags="${flags} $i"; g=flag;;
       *) p=$i
          if [ "$g" = run ]; then
             extra_params="${extra_params} $p"
             g=0
          else
             if [ "$others" = "" ]; then
                others="$p"
             else
                others="$others $p"
             fi
             g=0
          fi
       ;;
 esac
 if [ "$g" = flag ]; then
    flags="${flags}=True";
 fi
done
eval "${flags}"
eval "${vars}"

if [ -n "$mvar" ]; then
   main_vars="-mv ${main_vars}"
fi
onError=break
echo "==================== Reval.sh ======================"
echo "Flags:${flags}"
echo "Variables:${vars}"
echo "Run Extra Params:${extra_params}"
echo "Others:${others}"

arr=($others)
if [ ${#arr[@]} -lt 2 ]; then
   echo "Output folder and config file or patterns are required (eg. bash eval.sh out1 pat1 pat2 )"
   exit
fi
output=${arr[0]}
echo "Output: $output"
configs=${arr[@]:1} 
echo "Configs: $configs"

######################################## Task flags:
if [ -n "$_ttasks" ]; then
   _tasks="${_tasks}#qnli#rte#mrpc#qqp"
fi
if [ -n "$_gtasks" ]; then
   _tasks="${_tasks}cola#qqp#mrpc#mnli#qnli#rte#stsb#sst2"
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
   _tasks="${_tasks}#xAttr#xReact#xIntent#oReact#oEffect#oWant#xNeed#xEffect#xWant"
fi
if [ -n "$_satasks" ]; then
   _tasks="${_tasks}#xAttr#xIntent#xReact#xWant#oWant"
fi
if [ -n "$_ltasks2" ]; then
   _tasks="mnli#qnli#qqp#mrpc#imdb#sst2#superglue-boolq#stsb"
fi
if [ -z "$_tasks" ]; then
   echo "_tasks (target tasks) is missinge e.g. _tasks=mnli#qqp#rte or use tasks flags e.g _gtasks for all glue tasks "
   exit
fi
if [ -n "$_seqt" ] && [ -z "$_stasks" ]; then
   _stasks=$_tasks
fi
if [ -z "$_nsp" ] && [ -z "$_stasks" ] && [ -z "$_src" ]; then
   echo "_stasks (source tasks for source prompts) is missinge e.g. _stasks=mnli#qqp#rte  or use _nsp=4 to use 4 source prompts or use _seqt flag to use the target tasks as source tasks"
   exit
fi
if [ -z "$_nsp" ] && [ -z "$_ppx" ]; then
   if [ -z "$_lsp" ] || [ "$_lsp" = "True" ]; then
      echo "_ppx (prompts prefix of saved source prompts) is missinge e.g. _ppx pat or use _lsp=False if you don't load source prompts"
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
echo "Tasks: ===================="
echo $_tasks
if [ -z "$_single" ]; then
   _tasks=$(echo "$_tasks" | sed "s/\#/@/g")
   echo "Multi Tasks: $_tasks"
fi

####################################
#   Default variables
if [ -z "$_exp" ]; then _exp=self; fi # Experiment name
if [ -z "$_seed" ]; then _seed=123; fi # Experiment seed
if [ -z "$_json" ]; then  _json="exp.json"; fi  # The base config file
if [ -n "$_all" ]; then _json="json"; fi

if [ -z $_bs ]; then _bs=12; fi  # batch size
if [ -z $_ep ]; then _ep=10; fi  # epochs
if [ -z $_tn ]; then _tn=10; fi  # train number
if [ -n "$_all_test" ]; then _tsn=-1; fi  # number of test dataset
if [ -z "$_tsn" ]; then _tsn=100; fi

if [ "$_nsp" = "all" ]; then _nsp=${#_tasks[@]}; fi
if [ -z "$_nsp" ]; then  _nsp=0; fi # Number of source prompts

if [ -z "$_rpx" ]; then  _rpx="${_ep}${_tn}"; fi # router prefix
if [ -z "$_skip" ]; then  _skip=False; fi #skip generation prompt if it exists
if [ -z "$_lp" ]; then  _lp=True; fi  #load prompts
if [ -z "$_pdir" ]; then  _pdir=prompts; fi # directory to save prompts
if [ -z "$_ppx" ]; then  _ppx="${_ep}${_tn}"; fi # prefix for prompts to load
if [ -z "$_opx" ]; then  _opx="${_ppx}"; fi # prefix for prompts to save


###################################
params=""
params="${params} --prompt_encoders_dir=$_pdir"
params="${params} --skip_if_prompt_exists=$_skip"
params="${params} --prompts_prefix=$_ppx"
params="${params} --output_prompts_prefix=$_opx"
params="${params} --load_prompts=$_lp"
params="${params} --ignore_train_if_prompt_exists=True"
params="${params} --prompts_to_save=none"
params="${params} --save_router=$_sr"
params="${params} --save_source_prompts=True"
params="${params} --save_all_prompts=$_sp"
params="${params} --router_prefix=$_rpx"
params="${params} --use_saved_router=$_usr"
###################################
# Setting task Parameters
params="${params} --per_device_train_batch_size=$_bs"
params="${params} --per_device_eval_batch_size=$_bs"
params="${params} --@num_train_epochs=$_ep"

if [ -n "$_tasks" ]; then
   params="${params} --@task_name=$_tasks"
fi
params="${params} --@num_source_prompts=$_nsp"
params="${params} --@source_prompts=$_src"

if [ -n "$_cmm" ]; then
   if [ $_cmm = "cat" ]; then
      if [ -z "$_numt" ]; then  _numt=10; fi
      if [ -z "$_ntp" ]; then  _ntp=auto; fi # number of target prompts
   else
      if [ -z "$_numt" ]; then  _numt=50; fi
      if [ -z "$_ntp" ]; then  _ntp=0; fi 
   fi
   params="${params} --@compose_method=$_cmm"
   params="${params} --@num_prompt_tokens=$_numt"
   params="${params} --@num_target_prompts=$_ntp"
fi

params=$(echo "$params" | sed "s/ --/\n --/g")
extra_params=$(echo "$extra_params" | sed "s/ --/\n --/g")

if [ -z "$_path" ]; then _path=""; fi

logs=$HOME/logs/$output
for conf in "${configs[@]}"; do
   if [ -z "$_dir" ]; then
      files=$(find . -type f -name "*${conf}*" -path "*${_path}*")
   else
      files=$(find ${_dir} -type f -name "*${conf}*" -path "*${_path}*")
   fi
   for file in $files; do 
      echo $file
      filename=$(basename -- "$file")
      extension="${filename##*.}"
      filename="${filename%.*}"
      experiment=""
      if [ -z "$_train" ]; then
         params="${params} --reval"
      fi
      experiment=$filename
      if [ -n "$_pvf" ]; then
         echo "$file"
      elif [ -n "$_pv" ]; then
         echo "runat run -exp $experiment -cfg $file -lp $logs ${params} ${extra_params} ${main_vars}" 
      else
         echo "Training ..."
         runat run -cfg $file -exp $experiment -lp $logs ${params} ${extra_params} $main_vars 
      fi
      if [ -n "$_one" ]; then
         echo "Exit after one experiment"
         break 2
      fi
      if [ $? != 0 ] && [ "$onError" = "break" ]; then
         echo "exit 1"
         break 2
      fi
   done
done

echo "Finished!!!"

if [ -n "$_sd" ]; then
   echo "shut down"
   echo 'a' | sudo -S shutdown -h now
   exit
fi
