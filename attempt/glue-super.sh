alias runsh="sh ${HOME}/ATTEMPT/attempt/train.sh"
runsh _pt _temp=0-pcom _exp=gcom-1000 _tn=1000 _tsn=-1 _max _minus-gtasks _ppx=gcom-1000

runsh _pt _temp=0-pt _exp=glue-sep-noprompt-1000-full-test _sep-gtasks _tsn=-1 _bs=32 _tn=1000 

runsh _exp=pat-gcom1000-glue-sep _sep-gtasks _temp=0-pt _tn=1000 _attn=rb#const _ppx=pt  _src=gcom-1000_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat#concat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=-1 _bs=32 

runsh _exp=pat-gcom1000-glue-multi _gtasks _temp=0-pt _tn=1000 _attn=rb#const _ppx=pt  _src=gcom-1000_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat#concat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=-1 _bs=32

runsh _pt _temp=0-pt _exp=super-glue-sep-noprompt-1000-full-test _sep-sgtasks _max _tsn=-1 _tn=1000 

runsh _exp=pat-gcom1000-super-glue-sep _sep-sgtasks _temp=0-pt _tn=1000 _attn=rb#const _ppx=pt  _src=gcom-1000_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat#concat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=-1 _bs=32

runsh _exp=pat-gcom1000-super-glue-multi _sgtasks _temp=0-pt _tn=1000 _attn=rb#const _ppx=pt  _src=gcom-1000_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat#concat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=-1 _bs=32

