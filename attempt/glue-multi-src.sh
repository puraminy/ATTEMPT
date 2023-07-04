extra=""
#extra="-pv ex"
tsn=-1
while getopts p:t:n:m:s:c: flag
do
    case "${flag}" in
        n) num_data=${OPTARG};;
        c) cat=${OPTARG};;
        t) task=${OPTARG};;
        s) stage=${OPTARG};;
    esac
done

if [ -z $stage ]; then
   stage=2
fi
if [ -z $num_data ]; then
   num_data=500
fi
if [ -z $cat ]; then
   cat="glue-com-${num_data}"
fi

cont=1
while [ "$cont" -eq 1 ]; do
   cont=0
   case "$stage" in 
     *1*)
        train_src="true"
        stage=$(echo "$stage" | sed "s/1//")
        cont=1
       ;;
     *2*)
        train_multi="true"
        stage=$(echo "$stage" | sed "s/2//")
        cont=1
       ;;
     *3*)
        train_sep="true"
        stage=$(echo "$stage" | sed "s/3//")
        cont=1
       ;;
   esac
done
alias runsh="sh ${HOME}/ATTEMPT/attempt/train.sh"

if [ -z $train_src ]; then
   echo "No train src"
else
   echo "Train src"
   runsh \
      _cat=$cat _pt _temp=0-pcom \
      _exp=glue-${num_data} \
      _tn=$num_data \
      _tsn=$tsn \
      _max \
      _minus-gtasks \
      _ppx=glue-${num_data} $extra
fi 

#runsh _cat=$cat _pt _temp=0-pt _exp=glue-sep-noprompt-${num_data}-full-test _sep-gtasks _tsn=$tsn -bs=16 _tn=${num_data}  $extra

if [ -z $train_multi ]; then
   echo "No train multi"
else
   echo "Train Multi"
   runsh \
   _learn_sp=False \
   _cat=$cat \
   _exp=pat-glue${num_data}-glue-multi \
   _task=mnli \
   _temp=0-pt \
   _tn=${num_data} \
   _attn=rb#const \
   _ppx=pt  \
   _src="src-glue-${num_data}_1_mlp_qnli@src-glue-${num_data}_1_mlp_qqp@src-glue-${num_data}_1_mlp_rte" \
   --@load_source_prompts=True \
   --@use_private_prompts=True \
   --@compose_method=wavg \
   -mv task_name--learn_source_prompts--attn_method \
   _max \
   _tsn=$tsn \
   _bs=16 \
   --attend_to_all=True \
   $extra
fi

#runsh _cat=$cat _exp=pat-glue${num_data}-glue-sep _sep-gtasks _temp=0-pt _tn=${num_data} _attn=rb#const _ppx=pt  _src=glue-${num_data}_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=$tsn _bs=16  $extra
