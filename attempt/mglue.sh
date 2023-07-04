extra=""
#extra="-pv ex"
tsn=300
alias runsh="sh ${HOME}/ATTEMPT/attempt/train.sh"
#runsh _cat=glue503 _pt _temp=0-pt _exp=glue-500 _tn=500 _tsn=$tsn _max _sep-src-gtasks _ppx=src-glue-500 $extra

#runsh _cat=glue503 _pt _temp=0-pt _exp=glue-sep-noprompt-500-full-test _sep-gtasks _tsn=$tsn -bs=16 _tn=500  $extra

runsh _cat=glue503 _exp=pat-glue500-glue-multi _src-gtasks _gtasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=$tsn _bs=16 $extra

runsh _cat=glue503 _exp=pat-glue500-glue-sep _src-gtasks  _sep-gtasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=$tsn _bs=16  $extra
