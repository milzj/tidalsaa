date=$(date '+%d-%b-%Y-%H-%M-%S')

# mpiexec -n 2 python simulate_experiment.py $date
python simulate_experiment.py $date
