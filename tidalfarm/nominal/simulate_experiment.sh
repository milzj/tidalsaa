date=$(date '+%d-%b-%Y-%H-%M-%S')

# mpiexec -n 5 python simulate_experiment.py $date
python simulate_experiment.py $date
