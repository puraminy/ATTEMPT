alias runsh="sh ${HOME}/ATTEMPT/attempt/train.sh"
runsh _pt _temp=0-pt _exp=glue-sep-noprompt-500-full-test _sep-gtasks _tsn=-1 -bs=16 _tn=500 

runsh _exp=pat-gcom500-glue-sep _sep-gtasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  _src=gcom-500_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat#concat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=-1 _bs=16 

runsh _exp=pat-gcom500-glue-multi _gtasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  _src=gcom-500_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat#concat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=-1 _bs=16

