#!/usr/bin/bash
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
tn=20
ppx="${ep}${tn}"
# ppx=20200
nums="_ep $ep _tsn 100 _bs 12"

logs=$HOME/logs/$1
if [ $1 = "test" ]; then
   rm -rf $logs
   ep=5
   tn=10
else
   if [ $ep -lt 10 ]; then
      echo "epochs are too low!"
      exit
   fi
fi

for tn in 20 50; do
for seed in 123; do
for cmm in cat wavg; do
   if [ $cmm = "cat" ]; then
      numt=10
   else
      numt=50
   fi
ntp=0
for numt in 10; do
for nsp in 0 2; do
if [ $nsp -eq 0 ]; then
   src="_seqt"
else
   src=""
fi
#for tasks in "_tasks qnli stsb mnli qqp"; do 
#for tasks in _gtasks _atasks; do 
for tasks in _gtasks; do 
   catname="${1}$tasks-$cmm-$numt-$nsp-seed-$seed"
   common="${params} _tn $tn $tasks $src _numt $numt _ntp $ntp _nsp $nsp _prefix"
   mets="$common $nums _cmm $cmm "

   SIP_args="$mets _upp _lsp _ppx $ppx _learn_sp False "
   SIL_args="$mets _lsp _ppx $ppx"
   SILP_args="$mets _upp _lsp _ppx $ppx"
   SILPI_args="$mets _upp _lsp _ppx $ppx _lpp"
   SL_args="$mets _lsp False "
   ST_args="$mets _lsp False _addt True "
   SLP_args="$mets _upp _lsp False"
   SLPI_args="$mets _upp _lsp False _lpp _ppx $ppx"
   PI_args="$common _pt $tasks _upp _lpp _lsp False $nums"
   P_args="$common _pt $tasks $nums _skip"

   for met in P SLPI SLP SL SIP SIL SILP SILPI PI; do
   # for met in ST SL; do # SIP SIL SILP SILPI; do
   # for met in SL SLP; do
       echo $met
       if [[ "$met" == *SI* ]] && [ "$nsp" -ne 0 ]; then
          continue
       fi
       args_variable="${met}_args"
       if [ -n "${!args_variable}" ]; then
           bash train.sh "_cat $catname _exp ${met} ${!args_variable} _seed $seed -lp $logs"
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

if [[ "$*" =~ "_shutdown" ]]; then
   echo "shut down"
   echo 'a' | sudo -S shutdown -h now
   exit
fi
