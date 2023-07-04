#!/usr/bin/bash
# shopt -s expand_aliases 
# source ~/aa
####################
bash_params=""
global_run_params=""
extra_params=""
sep="#"
onError="break"
g=0
for i in $@
do
   case $i in
       _*) bash_params="${bash_params} $i"; g=bash;;
       g-*)
         i=${i#g}
         global_run_params="${global_run_params} $i";
         g=run
       ;;
      # Other Parameter 
       *) p=$i
       if [ "$g" = run ]
       then
          global_run_params="${global_run_params} $p"
          g=0
       elif [ "$g" = bash ]; then
          bash_params=${bash_params%=*}
          bash_params="${bash_params}=$i"
          g=arr
       elif [ "$g" = arr ]; then
          bash_params="${bash_params}${sep}$i"
          g=arr
       else
          others="$others $p"
       fi
       ;;
 esac
 if [ "$g" = bash ]; then
    bash_params="${bash_params}=1";
 fi
done

get_com_ppx() {
   arr=($1)
   IFS=$'\n' sorted=($(sort <<<"${arr[*]}")); unset IFS
   tlist=$(printf "%s-" "${sorted[@]}")
   tlist=${tlist%-}
   result=${tlist}
}
############################## Evaluate them
echo "Bash params: ${bash_params}"
echo "Global: $global_run_params"
eval "${bash_params}"

if [ -z "$_cat" ]; then
   echo "Please specify the category"
   exit
fi

if [ -z "$_sep" ]; then
   _multi="_multi"
else 
   _multi=""
fi
if [ -z "$_bs" ]; then
   _bs=20
fi
if [ -z "$_tn" ]; then
   _tn=100
fi
if [ -z "$_tsn" ]; then
   _tsn=100
fi
if [ -z "$_lr" ]; then  _lr=0.01; fi
if [ -z "$_alr" ]; then _alr=0.07; fi
if [ -z "$_temp" ]; then
   _temp=ptar
fi
if [ -z "$_ep" ]; then
   _ep=15
fi
if [ -z "$_nsp" ]; then
   _nsp=7
fi
if [ -z "$_inpx" ]; then
   _inpx=10100
fi
if [ -z "$_outpx" ]; then
   if [ -n "$_stn" ]; then
      _outpx="${_ep}${_stn}"
   else
      _outpx="${_ep}${_tn}"
   fi
fi
if [ -n "$_tasks" ]; then
   tasks=$_tasks
   tasks=$(echo "$tasks" | sed "s/${sep}/ /g")
   tasks="_tasks $tasks"
fi

if [ -n "$_gtasks" ]; then
   tasks="_gtasks" 
fi
if [ -n "$_ltasks" ]; then
   tasks="_ltasks"
fi
if [ -n "$_ltasks2" ]; then
   tasks="_ltasks2"
fi

if [ -n "$_otasks" ]; then
   tasks="_otasks" 
fi

if [ -n "$_sgtasks" ]; then
   tasks="_sgtasks" 
fi
if [ -n "$_atasks" ]; then
   tasks="_atasks"
fi
if [ -n "$_all" ]; then
   tasks="_tasks ${tasks} ${stasks}"
fi

if [ -n "$_stasks" ]; then
   stasks=$_stasks
   stasks=$(echo "$stasks" | sed "s/${sep}/ /g")
   stasks="_stasks $stasks"
else
   if [ -n "$_sstasks" ]; then
      stasks="_sstasks"
   elif [ -n "$_seqt" ]; then
      stasks=$tasks
   else
      stasks="_stasks wnli@mnli@qnli@sst2@squad@piqa@qqp@mrpc"
   fi
fi
if [ -z "$com_ppx" ]; then
   echo "ST: $stasks"
   get_com_ppx "$stasks" 
   com_ppx=${_outpx}_$result
   com_src="${com_ppx}_mlp_com"
   echo "Com ppx: $com_ppx" 
fi
if [ -n "$_seed" ]; then
   seed=(1 2 3 4)
else
   seed=123
fi

do_exp() {
   echo "do_exp $seed"
   for seed in "${seed[@]}"
   do
      case "$1" in 
         *_nsp*|*_spt*)
            run="_cat $_cat _seed $seed _ep $_ep _bs $_bs $1"
            ;;
         *_seqt*)
            run="_cat $_cat _seed $seed _ep $_ep _bs $_bs $1"
            ;;
         *)
            run="_cat $_cat _seed $seed $stasks _ep $_ep _bs $_bs $1" 
            ;;
      esac
      if [[ ${run} != *"_tasks"* ]]; then
         run="${run} $tasks"
      fi
      if [[ ${run} != *"_tn"* ]]; then
         run="${run} _tn $_tn"
      fi
      if [[ ${run} != *"_numt"* ]] && [ -n "$_numt" ]; then
         run="${run} _numt $_numt"
      fi
      if [[ ${run} != *"_tsn"* ]]; then
         run="${run} _tsn $_tsn"
      fi
      if [ -n "$_eval" ]; then
         run="${run} _eval"
      fi
      if [ -n "$_all_test" ]; then
         run="${run} _all_test"
      fi
      if [ -n "$_sp" ]; then
         run="${run} _sp"
      fi
      if [[ ${run} != *"_temp"* ]]; then
         run="${run} _temp $_temp"
      fi
      if [[ ${run} != *"_alr"* ]]; then
         run="${run} _alr $_alr"
      fi
      if [[ ${run} != *"_lr"* ]]; then
         run="${run} _lr $_lr"
      fi
      if [ -n "$_rep" ]; then
         run="${run} -rep"
      fi
      if [ -n "$_d" ]; then
         run="${run} -d $_d"
      fi
      if [ -n "$_pv" ]; then
         echo "run $run"
         echo "======================================================="
      elif [ -n "$_pvv" ]; then
         echo "run would be ${run}"
         if [ -n "$_pvv" ]; then
            bash train.sh $run _pv
         fi
      else
         if ! bash train.sh $run ; then
            echo "exit 1"
            exit 1
         fi
      fi
   done
}

if [ -n "$_stn" ]; then
   _inpx=$_outpx
   exp="_exp pts _tn $_stn _pt _temp pt _ppx $_outpx " 
   do_exp "$exp"
fi
if [ -n "$_pt" ]; then
   exp="_exp pt _pt _temp pt _ppx $_inpx _sp" 
   do_exp "$exp"
fi
#exp3="_exp ptcom _pt _temp pcom _ppx $com_ppx _tasks $stasks" 
if [ -n "$_sep_exp" ]; then
   exp="_exp sep _pat --learn_loaded_prompts=True --target_share=-1 --add_target=True _temp ptar _ppx $_inpx"
   do_exp "$exp"
fi

if [ -n "$_spt_sr" ]; then 
   exp="_exp spt _spt _sr $_multi _sp _ppx $_inpx"
   do_exp "$exp"
fi
if [ -n "$_nsp_sr" ]; then 
   exp="_exp nsp-sr _nsp $_nsp _sr $_multi _sp _ppx $_inpx"
   do_exp "$exp"
fi
if [ -n "$_seqt_sr" ]; then 
   exp="_exp seqsr _seqt _sr _multi _sp _ppx $_inpx"
   do_exp "$exp"
fi
if [ -n "$_seqt_usr_boot" ]; then
   if [ -z "$_seqt_sr" ]; then 
      exp="_exp seqsr _seqt _sr _multi _sp _ppx $_inpx"
      do_exp "$exp"
   fi
   spx="${_inpx}_mlp"
   for alr in 0.1 0.001; do
      exp="_exp seqt_usr-$alr _alr $alr _ep 10 _seqt _usr $_multi _rpx $_tn _sp _ppx $spx"
      do_exp "$exp"
      spx="${spx}_mlp"
   done
fi


cpx="$inpx"
if [ -n "$_boot" ]; then
   src=""
   for ((i=1; i<$_nsp; i++)); do
       src+="@com$i"
   done
   for ii in 1 2 3; do
      cpx="${cpx}_mlp"
      exp="_exp boot_$_nsp $_multi _src $src _ppx $cpx _sp"
      do_exp "$exp"
   done
fi
if [ -n "$_seqt" ]; then
   exp="_exp seqt _pat _seqt $_multi _ppx $_inpx _sp "
   do_exp "$exp"
fi
if [ -n "$_best_seqt" ]; then
   for at in "-1 5." "0 0.1"; do 
   for alr in 0.1 0.2; do 
   for lr in 0.05 0.1; do
      exp="_exp seqt _pat _adir ${at[0]} _tmpr ${at[1]} _alr $alr _lr $lr _seqt $_multi _ppx $_inpx _sp "
      do_exp "$exp"
   done
   done
   done
fi

spx=$_inpx
if [ -n "$_seqt_boot" ]; then
   for alr in 0.1 0.05 0.01; do
      exp="_exp sboot_$alr $_multi _alr $alr _seqt _ppx $spx _sp"
      do_exp "$exp"
      spx="${spx}_mlp"
   done
fi
spx=$_inpx
if [ -n "$_seqt_add" ]; then
   for ii in 1; do
      exp="_exp seqt-add4-$ii _alr 0.1 _lr 0.05 _pat _seqt _sp $_multi _ppx $spx --add_target=True --target_share=0.2"
      do_exp "$exp"
      spx="${spx}_mlp"
   done
fi

if [ -n "$_seqt_upp" ]; then
   exp="_exp seqt-upp-noat $_multi --learn_attention=False _pat _seqt --learn_loaded_prompts=True --use_private_prompts=True _temp ptar _ppx $_inpx --num_target_prompts=-1"
   do_exp "$exp"
   exp="_exp seqt-upp $_multi _alr 0.1 _lr 0.01 _pat _seqt --learn_loaded_prompts=True --use_private_prompts=True _temp ptar _ppx $_inpx --num_target_prompts=-1"
   do_exp "$exp"
fi

if [ -n "$_nsp_usr" ]; then
    base="_exp usr-base _ep 10 _nsp $_nsp $_multi --add_target=True --target_share=-1"
    do_exp "$base"
    exp="-rep _exp usr-noat --learn_attention=False _nsp $_nsp _ep 10 _usr $_multi --add_target=True --target_share=-1 _rpx $_tn _sp"
    do_exp "$exp"
    #exp="-rep _exp usr0.1 _nsp 7 _ep 10 _usr $_multi --add_target=True --target_share=-1 _rpx $_tn"
    #do_exp "$exp"
    #exp="-rep _exp usr0.01 _alr 0.01 _nsp 7 _ep 10 _usr $_multi --add_target=True --target_share=-1 _rpx $_tn"
    #do_exp "$exp"
    exp="-rep _exp usr0.01-0.1 _alr 0.01 _lr 0.1 _nsp $_nsp _ep 10 _usr $_multi --add_target=True --target_share=-1 _rpx $_tn"
    do_exp "$exp"
fi
if [ -n "$_nsp_upp" ]; then
   exp="_exp nsp-upp $_multi _pat _nsp $_nsp --learn_loaded_prompts=True --use_private_prompts=True _temp ptar _ppx $_inpx"
   do_exp "$exp"
fi
if [ -n "$_nsp_exp" ]; then
   for _tn in $_tn; do # 50 200 500 1000; do
      for _nsp in 5 7 8 9 10; do
         exp="_exp multi-nsp-$_nsp _tn $_tn $_multi _pat _nsp $_nsp --use_private_prompts=False _temp ptar _ppx $_inpx _sr --router_prefix=$_tn"
         do_exp "$exp"
      done
   done
fi
if [ -n "$_double" ]; then
   exp="_exp reg1 _multi _pat --use_private_prompts=True _temp ptar _ppx $_inpx"
   do_exp "$exp"
   temp=$tasks
   tasks=$stasks
   stasks=$temp
   exp="_exp reg2 _multi _pat --use_private_prompts=True _temp ptar _ppx $_inpx"
   do_exp "$exp"
fi
if [ -n "$_multi_exp" ]; then
   exp="_exp multi _multi _pat --learn_loaded_prompts=True --target_share=-1  --add_target=True _temp ptar _ppx $_inpx"
   do_exp "$exp"
fi
if [ -n "$_upp" ]; then
   exp="_exp upp-multi $_multi _pat --learn_loaded_prompts=True#False --use_private_prompts=True _temp ptar _ppx $_inpx"
   do_exp "$exp"
   #exp="_exp upp-sep _pat --learn_loaded_prompts=True#False --use_private_prompts=True _temp ptar _ppx $_inpx"
   #do_exp "$exp"
fi 
if [ -n "$_mcom" ]; then
   exp="_exp multi-com _pat $_multi _temp ptar _src $com_src -rep" 
   do_exp "$exp"
fi


