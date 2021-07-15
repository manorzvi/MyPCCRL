mkdir PccNs-v1_0
cd      PccNs-v1_0
python ../src/gym/stable_solve.py --env=PccNs-v1 --model-dir=./final_model/ &> out.out &
cd ../

mkdir PccNs-v8_0
cd      PccNs-v8_0
python ../src/gym/stable_solve.py --env=PccNs-v8 --model-dir=./final_model/ &> out.out &
cd ../

mkdir PccNs-v11_0
cd      PccNs-v11_0
python ../src/gym/stable_solve_2_senders.py --env=PccNs-v11  --model-dir=./final_model/ &> out.out &
cd ../

mkdir PccNs-v18_0
cd      PccNs-v18_0
python ../src/gym/stable_solve_2_senders.py --env=PccNs-v18  --model-dir=./final_model/ &> out.out &
cd ../

mkdir PccNs-v19_0
cd      PccNs-v19_0
python ../src/gym/stable_solve_2_senders.py --env=PccNs-v19  --model-dir=./final_model/ &> out.out &
cd ../

mkdir PccNs-v20_0
cd      PccNs-v20_0
python ../src/gym/stable_solve_2_senders.py --env=PccNs-v20  --model-dir=./final_model/ &> out.out &
cd ../