if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

for i in ENZYMES; do
  for j in gcn gin asap; do
    # python main.py --cuda $cuda --dataset $i --model $j
    for k in 10; do
      # python main.py --cuda $cuda --dataset $i --model $j
      python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
    done
  done
done

# gcn gin baselines
for i in MUTAG; do
  for j in ssgc; do
    for k in 0.2 0.3 0.4; do
      # echo 'pass'
      # python main.py --cuda $cuda --dataset $i --model $j
      # python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done

for i in DD; do
  for j in gin; do
    for k in 5 10 15 20; do
      echo 'pass'
      # python main.py --cuda $cuda --dataset $i --model $j
      # python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done

for i in Mutagenicity; do
  for j in gin_w; do
    for k in 10; do
      echo 'pass'
      # python main.py --cuda $cuda --dataset $i --model $j
      # python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done

for i in DD; do
  for j in sgc; do
    for k in 5 10 15 20; do
      echo 'pass'
      # python main.py --cuda $cuda --dataset $i --model $j
      # python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done

for i in DD; do
  for j in set2set_w; do
    # python main.py --cuda $cuda --dataset $i --model $j
    for k in 5 10 15 20; do
      # python main.py --cuda $cuda --dataset $i --model $j
      # python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
  done
 done
done
