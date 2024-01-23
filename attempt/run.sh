#!/usr/bin/bash
bash_params=""
global_run_params=""
extra_params=""
sep="#"
onError="continue"
g=0
for i in $@
do
   case $i in
       _*) bash_params="${bash_params} $i"; g=bash;;
       *) p=$i
          if [ "$g" = bash ]; then
             bash_params=${bash_params%=*}
             bash_params="${bash_params} $p"
          else
             others="$others $p"
          fi
       ;;
 esac
done

params="$*"
alias runsh="bash ${HOME}/ATTEMPT/attempt/train.sh"
onError=break
echo "Bash params: ${bash_params}"
echo "Global: $global_run_params"

ep=20
epp=20
tn=20
bs=12
ppx="${epp}${tn}"
# ppx=20200
nums="_ep $ep _tsn 100"

logs=$HOME/logs/$1
if [ $1 = "test" ]; then
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
for tn in 20; do
for seed in 123; do
for cmm in wavg cat; do
   if [ $cmm = "cat" ]; then
      numt=10
      ntp=0
   else
      numt=50
      ntp=0
   fi
for grm in "sign@rb@sigmoid@direct"; do
for soft in after nothing; do
for attn in rb; do 
for ntp in 5; do
for nrp in "3-5"; do
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
for tasks in _gtasks; do 
   ((ii++))
   catname="${1}$tasks-$cmm-$ntp-$nsp-seed-$seed-$route_method-$ii-$tn"
   common="${params} _nrp $nrp _attn $attn $nums _tst $tst _bs $bs _tn $tn $tasks $src _numt $numt _ntp $ntp _nsp $nsp _prefix"
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

   # for met in P SC SILP SL SLPI SLP SIP SIL SILPI; do
   for met in SILPI SL; do
   # for met in ST SL; do # SIP SIL SILP SILPI; do
   # for met in SC SLP; do
   # for met in SLPI SLP; do
   # for met in SLP SILPI SLPI SL; do
   # for met in SIPI SIP SILPI; do
       echo $met
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
           bash train.sh "_cat $catname _exp ${met} ${!args_variable} _seed $seed -lp $logs"
           if [[ "$*" =~ "_one" ]]; then
               echo "exit after first experiment"
               exit 0
           fi
           if [ $? != 0 ] && [ "$onError" = "break" ];
           then
               echo "exit 1"
               exit 1
           fi
       else
           echo "Variable ${args_variable} not defined for method ${met}."
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
done

if [[ "$*" =~ "_shutdown" ]]; then
   echo "shut down"
   echo 'a' | sudo -S shutdown -h now
   exit
fi
