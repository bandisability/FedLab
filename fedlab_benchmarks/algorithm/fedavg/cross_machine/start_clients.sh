#!/bin/bash

# start a group of client. continuous rank is required.
# example: bash start_clients.sh ip port wolrd_size 1 5 dataset
#           start client from rank 1-5

echo "Connecting server:($1:$2), world_size $3, rank $4-$5, dataset $6"


for ((i=$4; i<=$5; i++))
do
{
    echo "client ${i} started"
    python client.py --server_ip $1 --server_port $2 --world_size $3 --rank ${i} --dataset $6 --epoch 2 --ethernet $7
} & 
done
wait

