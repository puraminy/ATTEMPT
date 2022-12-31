
#!/bin/sh

g1=""
g2=""
for i in $@
do
   case $i in
       # -- option
       --*) g1="${g1} $i"; g=1;;
       
       -m) echo "------"; g=3;;
       # - option
       -*) g2="${g2} $i"; g=2;;
       
       # Parameter 
       *) p=$i
          if [ "$g" = 1 ]
          then
            g1="${g1} $p"
            g=0
          elif [ "$g" = 2 ]
          then
            g2="${g2} $p"
            g=0
          elif [ "$g" = 3 ]
          then
            m=$p 
            g=0
          else
            others="$others $p"
          fi
      ;;
   esac
done
echo "Others: ${others}"
config=$(echo $others | xargs)
model=t5-base
case "$HOME" in 
  *ahmad*)
    # Do stuff
    model=t5-base
    ;;
esac
if [ -z $config ]; then
   config=configs/baselines/prompt_tuning.json 
fi 
echo "Config: ${config}"
home=$HOME
alias runat="python3 ${home}/ATTEMPT/attempt/run_seq2seq.py"
# wrap experiments
folder=${PWD##*/}          

test=100
train=200
if [ -z $m ]; then
   m=11
fi
echo "m: ${m}"
if [ "$m" -eq "0" ]; then
  echo "testing train"
elif [ "$m" -eq "1" ]; then
  echo "testing train and test"
  config=configs/baselines/test.json 
fi
exp=att-xint3
log=${home}/logs   
echo "log: ${log}"
var="max_train_samples=200"
var="${var}--max_val_samples=10"
var="${var}--max_test_samples=100"
var="${var}--num_train_epochs=3#5"
echo "var: ${var}"

runat run $g2 -exp $exp -cfg $config -var ${var} 
case "$home" in 
  *content*)
    # Do stuff
	mv /content/*time*.log ${log}/$exp
	tar -czvf /content/${exp}-$m.tar.gz ${log}/$exp
	cp /content/${exp}-$m.tar.gz ${home}/logs 
    ;;
esac


