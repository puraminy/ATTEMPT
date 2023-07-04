#!/usr/bin/bash
shopt -s expand_aliases 
source ~/aa
#################  Get Bash Parameters
bash_params=""
global_run_params=""
extra_params=""
sep="@@@"
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
############################### Processing experiment parameters
get_params() {
   extra_params=""
   run_params=""
   echo "Extracting params from $1"
   params=($(echo -n $1 | sed "s/${sep}/ /g"))
   main_vars=""
   for i in "${params[@]}"
   do
      case $i in
         --*) 
            extra_params="${extra_params} $i"; 
          ;;
         -*) run_params="${run_params} $i"; g=run;;
      esac
   done
}
get_com_ppx() {
   arr=($(echo -n $1 | sed "s/${sep}/ /g"))
   IFS=$'\n' sorted=($(sort <<<"${arr[*]}")); unset IFS
   tlist=$(printf "%s-" "${sorted[@]}")
   tlist=${tlist%-}
   result=${tlist}
}
############################### Experiments

if [ -z $_tn ]; then
   echo "_tn is missinge (train samples)"
   exit
fi
if [ -z $_ptn ]; then
   _ptn=$_tn
fi
if [ -z $_tsn ]; then
   _tsn=200
fi
if [ -z $_numt ]; then
   _numt=40
fi
if [ -z $_temp ]; then
   _temp=0-pt
fi

if [ -z $_tasks ]; then
   echo "_tasks is missinge (target task or tasks) e.g. _tasks mnli rte "
   exit
fi
if [ -z $_cat ]; then
   echo "_cat (category) or a foder for experiments is missinge e.g. _cat nli "
   exit
fi

if [ -z $_ppx ]; then
   echo "_ppx (prompts prefix) is missinge e.g. _ppx nli "
   exit
fi


if [ -z $_pt ]; then
   echo "No source prompt tuning"
else
   echo "#-------------------------(Source) Prompt Tuning ---------------------"
   get_params $_pt

   sep_ppx=${_ppx}_${_temp}_${_tn}
   tasks=$(echo "$_tasks" | sed "s/${sep}/ /g")
   stasks=""
   if [ -n $_stasks ]; then
      echo "Source tasks are: $_stasks"
      arr=($(echo -n $_stasks | sed "s/@/ /g"))
      for t in "${arr[@]}"; do
         if [[ $tasks == *"$t"* ]]; then
            echo "$t already exists in tasks"
         else
            echo "adding $t"
            stasks="$stasks $t"
         fi
      done
   fi
   tasks="$stasks $tasks"
   echo "All training tasks are: $tasks"
   run="
      _pt _temp $_temp 
      _cat $_cat 
      _exp pt-sep-${_tn}-${_tsn} 
      _tn $_tn 
      _tsn $_tsn 
      _max 
      _tasks $tasks 
      _ppx $sep_ppx
      _lp True
      _bs 16
      --num_prompt_tokens=$_numt
      $extra_params 
      $global_run_params
      $run_params
      "
      echo "run would be ${run}"
   if [ -n "$_pv" ] || [ -n "$_pvv" ]; then
      echo "run would be ${run}"
      if [ -n "$_pvv" ]; then
         runit $run _pv
      fi
   else
      echo "Training..."
      runit $run || exit 1
      if [ $? != 0 ] && [ "$onError" = "break" ]; then
         echo "exit 1"
         exit
      fi
   fi
fi 

if [ -z $_ptm ]; then
   echo "Multi task source prompt tuning"
else
   echo "#--------------------------- Multi Task (Source) Prompt Tuning ------------------"
   if [ -z $_stasks ]; then
      echo "No source tasks was specified"
      exit
   else
      echo "Source tasks are: $_stasks"
      tasks="$_stasks"
   fi
   get_params $_ptm

   tasks=$(echo "$tasks" | sed "s/${sep}/ /g")
   get_com_ppx $_stasks  
   com_ppx=${_ppx}_${_tn}_$result

   run="
      _pt _multi _temp 0-pcom 
      _cat $_cat  
      _exp pt-multi-${_tn}-${_tsn}  
      _tasks $tasks  
      _tn ${_tn}   
      _tsn $_tsn  
      _ppx $com_ppx
      _bs 16
      --num_prompt_tokens=$_numt
      $extra_params 
      $global_run_params
      $run_params
      "
      echo "run would be ${run}"
   if [ -n "$_pv" ] || [ -n "$_pvv" ]; then
      echo "run would be ${run}"
      if [ -n "$_pvv" ]; then
         runit $run _pv
      fi
   else
      echo "Training..."
      runit $run || exit 1
      if [ $? != 0 ] && [ "$onError" = "break" ]; then
         echo "exit 1"
         exit
      fi
   fi
fi

echo "Using source prompts "

if [ -z $_pat ]; then
   echo "No task training with source prompts"
else
   echo "#-------------- Prompt Tuning using source prompts for $_tasks------------"

   if [ -z $com_ppx ]; then
      get_com_ppx $_stasks 
      com_ppx=${_ppx}_$result
   fi

   _com_src="${com_ppx}_mlp_com"
   if [ -z $_stasks ]; then
      echo "_stasks (source tasks) is missinge e.g. _stasks mnli qqp rte "
      exit
   fi

   stasks=$(echo "$_stasks" | sed "s/${sep}/\#/g")
   echo "Source tasks are: $stasks"
   arr=($(echo -n $_stasks | sed 's/@/ /g'))
   if [ -z $sep_ppx ]; then
      ppx=${_ppx} #_${_temp}
   else
      ppx=$sep_ppx
   fi
   _sep_src=""
   for t in "${arr[@]}"; do
      _sep_src="${_sep_src}@${ppx}_mlp_$t"
   done
   _sep_src=${_sep_src#"@"}
   echo "Used source prompts are: ${_sep_src}"

   get_params $_pat
   if [ -z $_tm ]; then 
      _tm=all_multi${sep}com_multi${sep}com_single
   fi
   arr=($(echo -n $_tm | sed "s/${sep}/ /g"))
   for method in "${arr[@]}" 
   do
      echo "# Prompt Tuning using source prompts for $_tasks using $method ------------"
      method_tasks=$(echo "$_tasks" | sed "s/$sep/ /g")
      multi=""
      if [ "$method" = com_multi ]; then 
         multi="_multi"
         _src=$_com_src
         echo "$method: Using $_src for $method_tasks "
      elif [ "$method" = com_single ]; then 
         _src=$_com_src
         echo "$method: Using $_src for $method_tasks "
      elif [ "$method" = all_multi ]; then 
         multi="_multi"
         _src=$_sep_src
         echo "$method: Using $_src for $method_tasks "
      elif [ "$method" = sep_sep ]; then 
         _src=$_sep_src
         echo "$method: Using $_src for $method_tasks "
      fi

      if [ -z $_src ]; then
         echo "No source prompts was provided"
         exit
      fi
      if [[ $_src != *"@"* ]]; then
         _src="${_src}@"
      fi
      run="
         $multi
         _cat ${_cat} 
         _exp pat-${method}-${_tn}-${_tsn} 
         _tasks ${method_tasks}
         _temp 0-pt
         _tn ${_ptn}
         _tsn $_tsn
         _src ${_src}
         _max
         $global_run_params
         ${run_params}
         --attend_to_all=True
         --num_prompt_tokens=$_numt
         ${extra_params}
         --method=$method
         "
      if [ -n "$_pv" ] || [ -n "$_pvv" ]; then
         echo "run would be ${run}"
         if [ -n "$_pvv" ]; then
            runit $run _pv
         fi
      else
         if ! runit $run ; then
            echo "exit 1"
            break
         fi
      fi
   done
fi

