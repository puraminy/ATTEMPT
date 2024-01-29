#!/usr/bin/bash
params=""
vars=""
flags=""
others=""
onError="continue"
debug=False
g=0
for i in $@
do
   if [[ "$i" == "-d" ]]; then
      debug=True
   fi
   case $i in
       --*) params="${params} $i"; g=param;;
       -*) params="${params} $i"; g=run;;
       *=*) vars="${vars} $i"; g=vars;;
       _*) flags="${flags} $i"; g=flag;;
       *) p=$i
          if [ "$g" = run ]; then
             params="${params} $p"
             g=0
          else
             if [ "$others" = "" ]; then
                others="$p"
             else
                others="$others $p"
             fi
             g=0
          fi
       ;;
 esac
 if [ "$g" = flag ]; then
    flags="${flags}=True";
 fi
done
eval "${flags}"
eval "${vars}"

alias runsh="bash ${HOME}/ATTEMPT/attempt/train.sh"
onError=break
echo "Flags:${flags}"
echo "Variables:${vars}"
echo "Run params:${params}"
echo "Others:${others}"
if [[ -n "$_sd" ]]; then
   echo "shut down after finish"
fi
arr=($others)

if [ ${#arr[@]} -lt 2 ]; then
   echo "Output folder and methods are required (eg. bash run.sh out1 method1 method2 ...)"
   exit
fi
output=${arr[0]}
echo "Output: $output"
methods=("${arr[@]:1}")
echo "Methods: $methods"


if [ "$methods" = "all" ]; then
   methods="SILPI SIL SLPI SILP SLP SL SIP"
fi

if [ -z $_ep ]; then
   _ep=10
fi
if [ -z $_epp ]; then
   _epp=20
fi
if [ -z $tnn ]; then
   _tnn=20
fi
if [ -z $_tn ]; then
  _tn=10
fi

bs=12
ppx="${_epp}${_tnn}"
# ppx=20200
nums="_ep $_ep _tsn 100"

logs=$HOME/logs/$output
if [ -n "$_test" ] || [ -n "$_debug" ]; then
   rm -rf $logs
   onError="break"
   _ep=5
   tn=10
   bs=4
else
   if [ $_ep -lt 10 ]; then
      echo "epochs are too low!"
      exit
   fi
fi
ii=0
if [ -z "$_nsp" ]; then
   _nsp=0
else
   _nsp=$(echo "$_nsp" | sed "s/\#/ /g")
fi
attn=rb
if [ -n "$_cmm" ]; then
   _cmm=$(echo "$_cmm" | sed "s/\#/ /g")
    echo "CMM: $_cmm"
else
   if [ -n "$_w" ]; then
      _cmm="wavg"
   fi
   if [ -n "$_wc" ]; then
      _cmm="wavg cat"
   fi
   if [ -n "$_cw" ]; then
      _cmm="cat wavg"
   fi
fi
if [ -z "$_tasks" ]; then
   _tasks="_tasks mnli qnli stsb qqp mrpc" 
fi

if [ -n "$_g" ]; then
   _tasks="_gtasks"
fi
if [ -n "$_a" ]; then
   _tasks="_atasks"
fi
if [ -n "$_sa" ]; then
   _tasks="_satasks"
fi


for tn in $_tn; do
for seed in 123; do
for gnm in "soft@"; do
for masking in "none" "0-col-1"; do
#for tasks in "_tasks mnli qnli qqp"; do 
#for route_method in bias ratt satt const direct; do
#for route_method in biasx biasp direct; do
#for tasks in _gtasks; do 
for nsp in $_nsp; do
echo "NSP: $nsp"
if [ $nsp -eq 0 ]; then
   src="_seqt"
else
   src=""
fi
for cmm in $_cmm; do
   echo "CMM: $cmm"
   ((ii++))
   catname="$cmm-$ntp-$nsp-seed-$seed-$ii-$tn"
   common="${flags} ${vars} ${params} _seed $seed _masking $masking $nums _bs $bs _tn $tn $_tasks $src _prefix"
   mets="$common _gnm $gnm _cmm $cmm "

   SIP_args="$mets _upp _lsp _ppx $ppx _learn_sp False "
   SI_args="$mets _lsp _ppx $ppx _learn_sp False "
   SIPI_args="$mets _upp _lsp _ppx $ppx _lpp _learn_sp False "
   SIL_args="$mets _lsp _ppx $ppx"
   SILP_args="$mets _upp _lsp _ppx $ppx"
   SILPI_args="$mets _upp _lsp _ppx $ppx _lpp"
   SL_args="$mets _lsp False "
   ST_args="$mets _lsp False _addt True "
   SLP_args="$mets _upp _lsp False"
   SLPI_args="$mets _upp _lsp False _lpp _ppx $ppx"
   PI_args="$common _pt $_tasks _upp _lpp _lsp False "
   P_args="_ep $_epp _tn $tnn _bs $bs _pt $_tasks _skip _prefix -merge"
   SC_args="$common _cmm $cmm _lsp False _rm const "

   for met in $methods; do
       if [[ "$met" == *SI* ]] && [ "$nsp" -ne 0 ]; then
          continue
       fi
   #   if [[ "$met" == *SL* ]]; then
   #      if [ "$route_method" != "biass" ]; then
   #         continue
   #      fi
   #   else
   #      if [ "$route_method" != "biasp" ]; then
   #         continue
   #      fi
   #   fi
       args_variable="${met}_args"
       if [ -n "${!args_variable}" ]; then
           echo $met
           if [ -n "$_rv" ]; then
              echo "_cat $catname _exp ${met} ${!args_variable} -lp $logs"
           else
              bash train.sh "_cat $catname _exp ${met} ${!args_variable} -lp $logs"
           fi
           if [[ -n "$_one" ]] || [[ -n "$_debug" ]]; then
               echo "exit after first experiment, use _all flag to run all loops!"
               exit 0
           fi
           if [ $? != 0 ] && [ "$onError" = "break" ];
           then
               echo "exit 1"
               exit 1
           fi
       else
           echo "Method ${args_variable} was not set for method ${met}."
           exit 0
       fi
   done
done
done
if [[ -z "$_all" ]]; then
   echo "exit after first loop"
   echo "exit after first all, use _all flag to run all loops!"
   exit 0
fi
done
done
done
done

if [ -d $logs ]; then
   cp run.sh $logs/
fi

if [[ -n "$_sd" ]]; then
   echo "shut down after finish"
   echo 'a' | sudo -S shutdown -h now
   exit
fi
