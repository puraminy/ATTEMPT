extra=""
# extra="-pv ex"
alias runsh="sh ${HOME}/ATTEMPT/attempt/train.sh"
#runsh _pt _cat=atomic500 _temp=0-pcom _exp=atomic-500 _tn=500 _tsn=200 _max _minus-atasks _ppx=atomic-500 $extra
#

#runsh _pt _cat=atomic500 _temp=0-pt _exp=atomic-sep-noprompt-500-full-test _sep-atasks _tsn=200 -bs=16 _tn=500  $extra

runsh _cat=atomic500 _exp=pat-atomic500-atomic-sep _sep-atasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  _src=atomic-500_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=200 _bs=16  $extra

runsh _cat=atomic500 _exp=pat-atomic500-atomic-multi _atasks _temp=0-pt _tn=500 _attn=rb#const _ppx=pt  _src=atomic-500_1_mlp_com@ --@load_source_prompts=True#False  --@use_private_prompts=True#False --@compose_method=wavg#cat -mv compose_method--task_name--load_source_prompts--use_private_prompts _max _tsn=200 _bs=16 $extra

