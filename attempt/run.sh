#!/usr/bin/bash
params=""
vars=""
flags=""
others=""
sep="#"
onError="continue"
g=0
for i in $@
do
   case $i in
       -*) params="${params} $i"; g=run;;
       *=*) vars="${vars} $i"; g=vars;;
       _*) flags="${flags} $i"; g=flag;;
       *) p=$i
          if [ "$g" = run ]; then
             params=${params%=*}
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
arr=($others)

if [ ${#arr[@]} -lt 2 ]; then
   echo "Output folder and methods are required (eg. bash run.sh out1 method1 method2 ...)"
   exit
fi
output=${arr[0]}
echo "Output: $output"
methods=("${arr[@]:1}")
echo "Methods: $methods"

ep=20
epp=20
tn=20
bs=12
ppx="${epp}${tn}"
# ppx=20200
nums="_ep $ep _tsn 100"

logs=$HOME/logs/$output
if [ -n "$_test" ]; then
   rm -rf $logs
   onError="break"
   ep=5
   tn=10
   bs=4
else
   if [ $ep -lt 10 ]; then
      echo "epochs are too low!"
      exit
   fi
fi
ii=0
nsp=0
tst=1
attn=rb
for tn in 20; do
for seed in 123; do
for cmm in cat wavg; do
   if [ $cmm = "cat" ]; then
      numt=10
      ntp=5
   else
      numt=50
      ntp=0
   fi
for grm in "sign@direct"; do
for attn in rb; do 
for soft in after nothing; do
for ntp in 5; do
for masking in "none" "0-col-1"; do
if [ $nsp -eq 0 ]; then
   src="_seqt"
else
   src=""
fi
#for tasks in "_tasks mnli qnli rte stsb qqp"; do 
#for tasks in "_tasks mnli qnli qqp"; do 
#for route_method in bias ratt satt const direct; do
#for route_method in biasx biasp direct; do
for route_method in direct; do
for thresh in 0.0 100; do
for tasks in _gtasks; do 
   ((ii++))
   catname="${1}$tasks-$cmm-$ntp-$nsp-seed-$seed-$route_method-$ii-$tn"
   common="${params} _thresh $thresh  _masking $masking _attn $attn $nums _tst $tst _bs $bs _tn $tn $tasks $src _numt $numt _ntp $ntp _nsp $nsp _prefix"
   mets="$common _soft $soft _grm $grm _cmm $cmm _rm $route_method"

   SIP_args="$mets _upp _lsp _ppx $ppx _learn_sp False "
   SIPI_args="$mets _upp _lsp _ppx $ppx _lpp _learn_sp False "
   SIL_args="$mets _lsp _ppx $ppx"
   SILP_args="$mets _upp _lsp _ppx $ppx"
   SILPI_args="$mets _upp _lsp _ppx $ppx _lpp"
   SL_args="$mets _lsp False "
   ST_args="$mets _lsp False _addt True "
   SLP_args="$mets _upp _lsp False"
   SLPI_args="$mets _upp _lsp False _lpp _ppx $ppx"
   PI_args="$common _pt $tasks _upp _lpp _lsp False "
   P_args="$common _pt $tasks _skip"
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
           if [ -n "$_pv" ]; then
              echo "_cat $catname _exp ${met} ${!args_variable} _seed $seed -lp $logs"
              echo "exit after preview rund command"
              exit 0
           fi
           bash train.sh "_cat $catname _exp ${met} ${!args_variable} _seed $seed -lp $logs"
           if [[ -n "$_one" ]]; then
               echo "exit after first experiment"
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
if [[ -n "$_loop1" ]]; then
   echo "exit after first loop"
   exit 0
fi
done
done
done
done
done
done
done
done
done
done

cp run.sh $logs

if [[ -n "$_shutdown" ]]; then
   echo "shut down after finish"
   echo 'a' | sudo -S shutdown -h now
   exit
fi
