alias barun="bash /content/ATTEMPT/attempt/run.sh"
alias abarun="bash /content/Asai/ATTEMPT/attempt/run.sh"
#alias retrain="barun _re --do_train=True"
alias procdf="python3 /content/ATTEMPT/attempt/procdf.py"
alias retrain="python3 /content/ATTEMPT/attempt/run_seq2seq.py run "

# Train Splitter

# retrain FT -to splitter -conf splitter --t=splitter _atomic --lr=0.0003 --save=splitter --tn=500 --ep=5 _rels 

# Apply Splitter
# reval FT -conf split-omcs --t=omcs --tsn=20000 --model=t5-base-splitter-500

# Apply Classifier
# reval FT -conf classify-omcs --t=free-rels --tsn=16000 --model=t5-base-classifier-500 --cache=False 

# Unsupervised Pre-Training using 8000 sample
# retrain FT --t=free-cs -to fc-2 --cache=False --tn=8000 --ep=5 --tsn=100 --bs=16 --temp=per_sample --lr=0.0001 --model=t5-base#t5-lm#t5-v1 --save=template-free
#procdf ft-map
#procdf pt-map
#procdf ft-filling
#procdf pt-filling


# Add key-value pairs to the array
##my_dict=(
#    ['sup']='Mapping'
#    ['unsup']='MaskedMapping'
#    ['sup-nat']='Prompting'
#    ['unsup-nat']='MaskedPrompting'
#    ['vnat-v3']='MaskedChoicePrompting'
#    ['vnat_1-vs2']='ChoicePrompting'
#    ['vnat_0-v4']='MaskedAnswerPrompting'
#    ['vnat_0-vs2']='AnswerPrompting'
#    ['0-ptar-sup']='PostPT'
#    ['ptar-sup']='PreSup'
#    ['ptar-unsup-nat']='MaskedPrePT'
#    ['ptar-sup-nat']='PrePT'
#    ['0-ptar-unsup']='MaskedPostPT'
#    ['0-ptar-vnat-v3']='MaskedAnswerPT'
#    ['0-ptar-vnat_1-vs1']='AnswerPT'
#    ['ptar-vnat_0-v4']='PreMaskedAnswerPT'
#    ['0-ptar-vnat_0-v4']='PostMaskedAnswerPT'
#    ['ptar-vnat_0-vs2']='PreAnswerPT'
#    ['0-ptar-vnat_0-vs2']='PostAnswerPT'
#)
#

if [ -z "$1" ]; then
   echo "Provide parameter 1"
   exit
fi

t=$1
ns="1"
for tune in FT; do
   echo "Tune: $tune"
for mod in large; do
   echo "mod: $mod"
#for met in none; do
for met in none sup unsup; do
   echo "met: $met"
#for version in free-8000; do
for version in opsent-6500 rand-8000 free-8000; do
   echo "version: $version"
if [ $mod = "large" ]; then
   tbs=10
else
   tbs=10
fi

if [ $met = "none" ]; then
   model="t5-$mod"
else
   model="t5-$mod-$met-$version" 
fi
echo "$model"
if [ $t = "none" ]; then
   folder=$mod
else
   folder=$t
fi
   echo "model: $model "
if [ $tune = "FT" ]; then
#if [ "$tune" = "FT" ] && { [ "$met" = "none" ] ||  [ "$met" = "unsup" ] || [ "$met" = "mixed" ]; }; then
   for task in obqa; do
   echo "task: $task"
   echo "----------------------"
   retrain FT -to ft-$folder-$mod-$task --tn=30 --tsn=-1 --d=123 --ep=3 \
      --comment-tn=9741#4950 \
      --chpos="start" \
      --qpos="end" \
      --c-t="xNeed#xAttr#xIntent#AtLocation" \
      --c-t="xWant#xIntent#xNeed#xAttr#AtLocation#ObjectUse" \
      --c-t="AtLocation#ObjectUse#xAttr#xIntent#xNeed" \
      --c-t="HasProperty#ObjectUse#xWant#oWant#xEffect#xReact#oReact#oEffect#Causes#MadeUpOf#CapableOf#isBefore#isAfter#HinderedBy#Desires#HasSubEvent#NotDesires" \
      --t=$task \
      --comment-t="xNeed#xIntent#xWant" \
      --omit="fact1" \
      --c-temp="unsup-nat#vnat_0-vs2#vnat_0-v4" \
      --c-temp="sup-nat#vnat_1-v4#vnat_1-vs2" \
      --c-temp="unsup-nat#sup-nat" \
      --c-temp="sup-nat#unsup-nat" \
      --c-temp="vnat_1-vs2#vnat_1-vs2-tr2" \
      --c-temp="vnat_10-vs1#vnat_0-v4#vnat-v3#sup#unsup-nat" \
      --temp="vnat_0-v4#vnat-v3#vnat_1-vs2#vnat_0-vs2" \
      --comment-temp="vnat_0-vs2#vnat_1-vs2#vnat-v3#vnat_0-v4#sup#unsup#sup-nat#unsup-nat"  \
      --c-temp="vnat-v3#vnat_0-v4"  \
      --comment-op-temp="vnat_1-vs2#vnat_0-vs2"  \
      --comment="#vnat_1-vs1#unsup-nat" \
      --commenttemp="vnat-v3#vnat_1-vs2#temp_len#sup#vnat_1-vs1#sup#sup-nat#unsup#unsup-nat" \
      -skip \
      --lr=0.0001 \
      --model=$model \
      --bs=4 --tbs=$tbs --opt_type=ada --sph=3 --cache=True -merge=task_name  -last=task_name
   done
else
   for task in "commonsense-qa"; do
      if [ "$task" = "commonsense-qa" ]; then
         tn="30#425#1700"
      else
         tn="30#298#991"
      fi
      if [ "$tn" = "30" ]; then
         seed="123#45#67"
      else
         seed="123"
      fi
      retrain PT -to pt-$folder --tn=30 --tsn=100 --d=123  \
      --c-t="xAttr#AtLocation#CapableOf#HasProperty#ObjectUse#isFilledBy" \
      --t="xNeed#xIntent#xWant" \
      --temp="unsup-nat#sup-nat#0-ptar-unsup#0-ptar-sup" \
      --comment-temp="ptar-vnat_0-vs2" \
      --comment-temp="0-ptar-sup" \
      --comment-temp="ptar-vnat_0-v4#ptar-vnat_0-vs2" \
      --comment-temp="ptar-vnat_0-v4#ptar-vnat_0-vs2" \
      --comment-cs-ptar-temp="ptar-vnat_0-v4#ptar-vnat-v3" \
      --comment-cs-temp="0-ptar-vnat_0-v4#0-ptar-vnat-v3" \
      --comment-op-temp="0-ptar-vnat_0-vs2#0-ptar-vnat_1-vs2" \
      --comment-op-ptar-temp="ptar-vnat_0-vs2#ptar-vnat_1-vs2" \
      --comment="0-ptar-vnat-v3#0-ptar-vnat_1-vs1#0-ptar-sup#ptar-unsup-nat#ptar-sup-nat#0-ptar-unsup" \
      --comment="0-ptar-vnat_1-vs1#0-ptar-vnat-v3#ptar-unsup-nat#ptar-sup" \
      --ep=12 -skip \
      --omit="fact1" \
      --numt=10 \
      --model=$model \
      --lr=0.08 --bs=32  \
      --opt_type=ada --enctype=mlp_res --sph=3 \
      --cache=True -merge=task_name -last=task_name 
   done
fi
done
done
done
done

if [ "$2" = "sd" ]; then
   echo 'a' | sudo -S shutdown -h now
fi

exit

--temp="vnat_10-v3#vnat_10-v2#vnat_10-v1#sup#unsup#unsup-nat" \
retrain FT --t=free-cs -to fc-2 --cache=False --tn=12000 --ep=3 --tsn=100 --bs=8 --temp=unsup#sup --lr=0.0001 --model=t5-v1#t5-lm#t5-base --save=template-sent2


--temp="vnat_1-vs1#sup-nat#unsup-nat#sup#unsup#vnat-v3#vnat-vs2#vnat_1-vs2#vnat-vs1" --comment="#sup#unsup#unsup-nat" \
exit



#
#for model in v1 base lm; do
retrain FT -to ft-clt --tn=30 _clt --temp="sup-nat#unsup-nat#sup#unsup" \
#   --ep=5 \
#   --lr=0.0005 \
#   --model="t5-$model" \
#   --bs=8  --opt_type=ada  --sph=3 --cache=True _single -merge=task_name 
#done 
#

#for met in sup unsup per_sample; do
#for model in v1 base lm; do
#retrain PT -to pt-clt --tn=30 _clt  \
#   --temp="ptar-unsup-nat#ptar-sup-nat#sup#unsup" \
#   --ep=10 \
#   --model=t5-$model-$met-free-8000 \
#   --lr=0.08 --bs=4  \
#   --opt_type=ada --enctype=mlp_res --sph=3 --cache=True _single -merge=task_name 
#done
#done

#for met in sup unsup per_sample; do
for model in v1 base; do
retrain PT -to pt-clt --tn=30 _clt  \
   --temp="ptar-unsup-nat#ptar-sup-nat#sup#unsup" \
   --ep=10 \
   --model=t5-$model \
   --lr=0.1 --bs=4  \
   --opt_type=ada --enctype=mlp_res --sph=3 --cache=True _single -merge=task_name 
done
#done

#echo 'a' | sudo -S shutdown -h now
exit
--model=t5-$model-$met-free-8000 \

#################

for met in sup unsup; do
for model in v1 base lm; do
retrain FT -to ft-33 --tn=30 _pht --temp="sup-nat#unsup-nat#sup#unsup" \
   --ep=5 \
   --lr=0.0005 \
   --model="t5-$model-$met-free-8000" \
   --bs=16  --opt_type=ada  --sph=3 --cache=True _single -merge=task_name 
done 
done

exit


for met in per_sample; do
for model in v1 base lm; do
retrain FT -to ft-44 --tn=30 _ust --temp="sup-nat#unsup-nat#sup#unsup" \
   --ep=5 \
   --lr=0.0005 \
   --model="t5-$model-$met-free-8000" \
   --bs=16  --opt_type=ada  --sph=3 --cache=True _single -merge=task_name 
done 
done


exit


if [ "$1" = "sd" ]; then
   echo 'a' | sudo -S shutdown -h now
fi
