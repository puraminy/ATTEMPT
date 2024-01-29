#!/usr/bin/bash
shopt -s expand_aliases 
source ~/aa

extra_params=""
run_params=""
bash_params=""
main_vars=""
for i in $@
do
   case $i in
      --*) 
            q=${i#"--"}
            extra_params="${extra_params} --@${q}"; 
            p=${i%=*}
            main_vars="${main_vars}${p}"
            g=extra
       ;;
      _*) bash_params="${bash_params} $i"; g=bash;;
      -*) run_params="${run_params} $i"; g=run;;
       # Other Parameter 
       *) p=$i
       if [ "$g" = run ]
       then
          run_params="${run_params} $p"
          g=0
       elif [ "$g" = bash ]; then
          bash_params=${bash_params%=*}
          bash_params="${bash_params}=$i" 
          g=arr
       elif [ "$g" = arr ]; then
          bash_params="${bash_params}#$i" 
          g=arr
       else
          others="$others $p"
       fi
       ;;
 esac
 if [ "$g" = bash ]; then
    bash_params="${bash_params}=True";
 fi
done
main_vars=${main_vars#"--"}
main_vars="${main_vars}--task_name"

echo "==================== Reval.sh ======================"
echo "Main experiment variables: $main_vars"
echo "Bash Prarams: ${bash_params}"
echo "Extra Prarams: ${extra_params}"
echo "Run Prarams: ${run_params}"
echo "Others: ${others}"
eval "${bash_params}"

main_params=""
if [ -z "$_exp" ]; then
   _exp=self
fi
if [ -z "$_seed" ]; then
   seed=123
else
   seed=$_seed
fi
#if [ -z "$_pat" ]; then
#   _pat="Eval"
#fi
if [ -n "$_all_test" ]; then
   _tsn=-1 
fi

if [ -z "$_json" ]; then
   _json="exp.json"
fi
if [ -z "$_tsn" ]; then
   tsn=100
else
   tsn=$_tsn
fi
if [ -z "$_files" ]; then
   files=$(find . -type f -name "*${_json}*" -path "*${_pat}*")
fi
echo "files: $files"
for file in $files; do 
   echo $file
   filename=$(basename -- "$file")
   extension="${filename##*.}"
   filename="${filename%.*}"
   experiment=""
   if [ -z "$_train" ]; then
      params="${params} --reval"
   else
      if [[ $_exp != "self" ]]; then 
         experiment="-exp $_exp/$filename"
      else
         experiment="-exp self" 
      fi
   fi
   if [ -n "$_pvf" ]; then
      echo "$file"
   elif [ -n "$_pv" ]; then
      echo "runat run ${run_params} $experiment -cfg $file ${params} $extra_params" 
   else
      echo "Training ..."
      runat run ${run_params} -cfg $file ${params} ${extra_params} 
   fi
   if [ -n "$_one" ]; then
      echo "Exit after one experiment"
      exit 0
   fi
   if [ $? != 0 ] && [ "$onError" = "break" ]; then
      echo "exit 1"
      exit 1
   fi
done


if [[ "$*" =~ "_shutdown" ]]; then
   echo "shut down"
   echo 'a' | sudo -S shutdown -h now
   exit
fi
