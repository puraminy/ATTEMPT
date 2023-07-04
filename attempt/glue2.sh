extra=""
#extra="-pv ex"
tsn=300
alias runsh="sh ${HOME}/ATTEMPT/attempt/train.sh"
#runsh _cat=glue-com-500 _pt _temp=0-pcom _exp=glue-500 _tn=500 _tsn=$tsn _max _minus-gtasks _ppx=glue-500 $extra

#runsh _cat=glue-com-500 _pt _temp=0-pt _exp=glue-sep-noprompt-500-full-test _sep-gtasks _tsn=$tsn -bs=16 _tn=500  $extra

runsh _cat=glue-com-502-cola6 _exp=pat-glue500-glue-multi _task=cola _temp=0-pt _tn=500 _attn=const _ppx=pt  _src=glue-500_1_mlp_com@ --@load_source_prompts=True  --@use_private_prompts=True --@compose_method=wavg -mv apply_softmax_to--anneal_dir--attn_method --anneal_dir=0  _max _tsn=$tsn _bs=16 $extra

#runsh _cat=glue-com-500 _exp=pat-glue500-glue-sep _sep-gtasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  _src=glue-500_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=$tsn _bs=16  $extra
