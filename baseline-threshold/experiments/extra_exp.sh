if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

# gcn gin baselines
for i in MUTAG PTC_MR PROTEINS DD ENZYMES; do
  for j in gcn_w; do
    for k in 5 10 15 20; do
      # echo 'pass'
      python main.py --cuda $cuda --dataset $i --model $j
      python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done

for i in MUTAG PTC_MR PROTEINS DD ENZYMES; do
  for j in sgc; do
    for k in 5 10 15 20; do
      echo 'pass'
      # python main.py --cuda $cuda --dataset $i --model $j --K 3
      # python main.py --cuda $cuda --dataset $i --model $j --K 3 --pinv True --topk $k
  done
 done
done

for i in MUTAG PTC_MR PROTEINS DD ENZYMES; do
  for j in asap; do
    for k in 5 10 15 20; do
      echo 'pass'
      # python main.py --cuda $cuda --dataset $i --model $j
      # python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done
