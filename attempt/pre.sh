alias procdf="python3 /content/ATTEMPT/attempt/procdf.py"
alias retrain="python3 /content/ATTEMPT/attempt/run_seq2seq.py run "
alias reval="python3 /content/ATTEMPT/attempt/run_seq2seq.py run -reval "

# Train Splitter

#retrain FT -to splitter -conf splitter --t=splitter  --lr=0.0003 --save=splitter --tn=100 --ep=5 --rels=all 

# Apply Splitter
#reval FT -conf split-omcs --t=omcs --tsn=20000 --model=t5-base-splitter-500
#reval FT -conf split-omcs --t=opsent --tsn=8000 --model=t5-base-splitter-300

# Apply Classifier
#reval FT -conf classify-omcs --t=free-rels --tsn=16000 --model=t5-base-classifier-500 --cache=False 

# Unsupervised Pre-Training using 8000 sample
for mod in t5-large; do
#retrain FT --t=free-cs -to opsent --cache=False --tn=6500 --ep=3 --tsn=100 --bs=16 --temp="sup#unsup" --lr=0.0001 --model=$mod --save=template-opsent-6500 -mod

retrain FT --t=free-cs -to fc-2 --cache=False --tn=8000 --ep=3 --tsn=100 --bs=24 --temp="unsup#sup#mixed" --lr=0.0001 --model=t5-large --save=template-8000-rand -mod 
done

#retrain FT --t=free-cs -to fc-2 --cache=False --tn=8000 --ep=5 --tsn=100 --bs=8 --temp=mixed#sup --lr=0.0001 --model=t5-calm --save=template-free --do_test=False


if [ "$1" = "sd" ]; then
   echo 'a' | sudo -S shutdown -h now
fi

exit
#procdf ft-map
#procdf pt-map
#procdf ft-filling
#procdf pt-filling

if [ -z "$1" ]; then
   echo "Provide parameter 1"
   exit
fi

t=$1
for tsk in "imdb#tweet-eval"; do 
for tune in FT; do
for mod in base lm v1; do
for met in  unsup mixed none sup; do # per_sample sup none; do
for version in sent2-12000; do
case $tsk in 
   _*)
          # Your code here
       task=$tsk
       tt="${tsk#_}"
       echo "Variable 'task' starts with an underscore: $task"
       ;;
   *)
       task="--t=$tsk"
       tt=sp
       ;;
esac

if [ $met = "none" ]; then
   model="t5-$mod"
else
   model="t5-$mod-$met-$version" 
fi
echo "$model"
if [ $t = "none" ]; then
   folder=$tt
else
   folder=$tt-$t
fi
if [ $tune = "FT" ]; then
   echo $task
   retrain FT -to ft-$folder --tn=30 --d=123 $task \
      --temp="vnat_10-v2#sup#unsup#unsup-nat" \
      --ep=5 -skip \
      --lr=0.0005 \
      --model=$model \
      --bs=4  --opt_type=ada --sph=3 --cache=True _single -merge=task_name  -last=task_name
   exit
else
   retrain PT -to pt-$folder --tn=30 $task  \
   --temp="ptar-unsup-vnat-$1#ptar-unsup-nat" \
   --ep=15 -skip \
   --model=$model \
   --lr=0.1 --bs=8  \
   --opt_type=ada --enctype=mlp_res --sph=3 \
   --cache=True _single -merge=task_name -last=task_name 
fi
done 
done
done
done
done


exit

retrain FT --t=free-cs -to fc-2 --cache=False --tn=12000 --ep=3 --tsn=100 --bs=8 --temp=unsup#sup --lr=0.0001 --model=t5-v1#t5-lm#t5-base --save=template-sent2


if [ "$1" = "sd" ]; then
   echo 'a' | sudo -S shutdown -h now
fi
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
