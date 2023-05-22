date=$(date '+%d-%b-%Y-%H-%M-%S')
source problem_data.sh
python simulate_risk_neutral.py $date $m $a $b $std $loc
