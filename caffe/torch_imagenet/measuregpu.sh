#!~/miniconda3/envs/py36/bin python
#!/bin/bash

measure(){
    batch_sizes=(16 32 64 128 256 512)
    num_workers=(2 4 8 16 32 64)
    domeasure ${#batch_sizes[*]} $(echo ${batch_sizes[*]}) ${#num_workers[*]} $(echo ${num_workers[*]})

}

measurehost12(){
    batch_sizes=(16 32 64 128)
    num_workers=(2 4 8 16 32 64)
    domeasure ${#batch_sizes[*]} $(echo ${batch_sizes[*]}) ${#num_workers[*]} $(echo ${num_workers[*]})
}

measurehost145(){
    #batch_sizes=(16 32 64 128)
    #num_workers=(2 4 8 16 32 64)
    #domeasure ${#batch_sizes[*]} $(echo ${batch_sizes[*]}) ${#num_workers[*]} $(echo ${num_workers[*]})

    batch_sizes=(16 32 64 128)
    num_workers=(2 4 8 16 32)
    domeasure ${#batch_sizes[*]} $(echo ${batch_sizes[*]}) ${#num_workers[*]} $(echo ${num_workers[*]})
}

measurehost143(){
    batch_sizes=(16 32 64 128 256 512)
    num_workers=(2 4 8 16 32 64)
    domeasure ${#batch_sizes[*]} $(echo ${batch_sizes[*]}) ${#num_workers[*]} $(echo ${num_workers[*]})
}

measuremgd189(){
    batch_sizes=(8 16 32 64 128)
    num_workers=(2 4 8 16)
    domeasure ${#batch_sizes[*]} $(echo ${batch_sizes[*]}) ${#num_workers[*]} $(echo ${num_workers[*]})
}

domeasure(){
    local origarray
    local len_batchs
    local len_workers
    local length=$[ $# ]

    origarray=($(echo "$@"))
    len_batchs=${origarray[0]}
    #index_workers=${origarray[`expr $len_batchs + 1`]}
    index_workers=`expr $len_batchs + 1`
    for (( i=1; $i<=$len_batchs; i++ )){
        for (( j=`expr $index_workers + 1`; $j<$length; j++ )){
            echo "i : $i, j: $j"
            echo " batch: ${origarray[$i]}, workers: ${origarray[$j]} "
            python main.py -a $net --measure $netdata -b ${origarray[$i]} -j ${origarray[$j]} --gpu 0 --lr 0.05 --weight-decay 0.00001 --epochs 95 --kind 000 $txt
        }
    }
}

domeasure2(){
    inti=0
    while(( $inti<${#batch_sizes[*]} ))
    do
        intj=0
        while(( $intj<${#num_workers[*]} ))
        do
            python main.py -a $net --measure $netdata -b ${batch_sizes[$inti]} -j ${num_workers[$intj]} --gpu 0 --lr 0.05 --weight-decay 0.00001 --epochs 95 --kind 000 $txt
            echo "i : $inti, j: $intj"
            echo " batch: ${batch_sizes[$inti]}, workers: ${num_workers[$intj]} "
            let "intj++"
        done
        let "inti++"
    done
}

while getopts 'h:n:' OPT; do
    case $OPT in
        h)
            host="$OPTARG";;
        n)
            net="$OPTARG"
            echo $net
            netdata=$net'mes'
            ;;
        ?)
            echo "Usage: `basename $0` [options] filename"
    esac
done
shift $(($OPTIND - 1))

echo " ==================run on host:"
echo $host
echo " ==================run on net:"
echo $net

case $host in
    gpuhome)
        echo " ==================run on host gpuhome:"
        echo $host
        txt="/home/datasets/imagenet/imagenet_hdf5"
        measure
        ;;
    host12)
        echo " ==================run on host12:"
        echo $host
        txt="/data/03/imagenet/imagenet_hdf5"
        measurehost12
        ;;
    host143)
        echo " ==================run on host143:"
        echo $host
        txt="/home/hpcl/data/imagenet/imagenet_hdf5"
        measurehost143
        ;;
    host145)
        echo " ==================run on host145:"
        echo $host
        txt="/media/disk2/data2/imagenet/imagenet_hdf5"
        measurehost145
        ;;
    mgd189)
        echo " ==================run on mgd189:"
        echo $host
        txt="/home/mgd/data/imagenet_hdf5_224"
        measuremgd189
        ;;
esac



