if [ -z "$1" ]; then
  echo "empty cuda input!"
  cuda=0
else
  cuda=$1
fi

for i in MUTAG PTC_MR PROTEINS NCI1 NCI109 ENZYMES Mutagenicity FRANKENSTEIN IMDB-BINARY IMDB-MULTI; do
  for j in gcn gcn_w gin gin_w asap asap_w gat graphsage graphsage_w set2set set2set_w ssgc; do
    # python main.py --cuda $cuda --dataset $i --model $j
    for k in 0.1 0.2 0.3 0.4; do
      python main.py --cuda $cuda --dataset $i --model $j
      python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k
    done
  done
done

for i in DD; do
  for j in gcn gcn_w gin gin_w asap asap_w gat graphsage graphsage_w set2set set2set_w ssgc; do
    # python main.py --cuda $cuda --dataset $i --model $j
    for k in 0.1 0.2 0.3 0.4; do
      python main.py --cuda $cuda --dataset $i --model $j --nhid 32 --batch_size 10
      python main.py --cuda $cuda --dataset $i --model $j --pinv True --topk $k --nhid 32 --batch_size 10
    done
  done
done



# for i in MUTAG PTC_MR DD PROTEINS NCI1 NCI109  ENZYMES Mutagenicity FRANKENSTEIN IMDB-BINARY IMDB-MULTI COLLAB REDDIT-BINARY

# do
# 	for j in gcn gin
# 	do
# 		python main.py  --cuda $1 --dataset $i --model $j
# 	done

# done
