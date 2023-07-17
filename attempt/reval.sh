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

echo "==================== Train.sh ======================"
echo "Main experiment variables: $main_vars"
echo "Bash Prarams: ${bash_params}"
echo "Extra Prarams: ${extra_params}"
echo "Run Prarams: ${run_params}"
eval "${bash_params}"

main_params=""
if [ -z "$_seed" ]; then
   seed=123
else
   seed=$_seed
fi
if [ -n "$_all_test" ]; then
   _tsn=-1 
fi
if [ -z "$_tsn" ]; then
   tsn=100
else
   tsn=$_tsn
fi
if [ -z "$_files" ]; then
   readarray -t files < <(find "$PWD" -type f -name "exp.json")
else
   files=$_tsn
fi
for file in $(find "$PWD" -type f -name "exp.json"); do
   echo $file
   for data_seed in $seed; do
      for test_num in $tsn; do
         params="${main_params} --data_seed=$data_seed"
         params="${params} --max_test_samples=$test_num"
         if [ -n "$_test" ] || [ -n "$_all_test" ]; then
            params="${params} --reval"
         fi
         if [ -n "$_pv" ]; then
            echo "runat run ${run_params} -cfg $file ${params} $extra_params" 
         else
            echo "Training ..."
            runat run ${run_params} -cfg $file ${params} ${extra_params} 
         fi
         if [ $? != 0 ] && [ "$onError" = "break" ]; then
            echo "exit 1"
            exit 1
         fi
      done
   done
done
