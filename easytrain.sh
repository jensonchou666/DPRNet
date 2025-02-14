#! /bin/bash

<< EOF
说明：
用此脚本快速开始训练模型, 对train.py的包装

<1>
------------------------------------------------------------------------------
easytrain config/net.py -c config/datasets/GID_Water.py {other_args}
------------------------------------------------------------------------------
等价于： python train.py config/net.py -c config/datasets/GID_Water.py {other_args}
------------------------------------------------------------------------------

<2>
------------------------------------------------------------------------------
easytrain 2 config/net.py -c config/datasets/GID_Water.py {other_args}
------------------------------------------------------------------------------
等价于： CUDA_VISIBLE_DEVICES=2 python train.py config/net.py 
            -c config/datasets/GID_Water.py {other_args}
指定CUDA
------------------------------------------------------------------------------

<3>
------------------------------------------------------------------------------
easytrain @ 3 config/net.py -c config/datasets/GID_Water.py {other_args}  &
------------------------------------------------------------------------------
等价于： CUDA_VISIBLE_DEVICES=3 python train.py config/net.py 
            -c config/datasets/GID_Water.py
                {other_args} > "work_dir/STD_LOG/last-3.txt" 2>&1 &
用于在后台执行，输出和错误信息保存到文件
------------------------------------------------------------------------------


EOF



OutDir=$(dirname "$0")/work_dir/STD_LOG
Train=$(dirname "$0")/train.py

CDX='@'

if [ $1 == '-h' ];then
    echo "easytrain [-h/$CDX] [GPU_IDS] ARGS"
    # echo "eg. easytrain 0,1 config/a.py config/ade20k.py"
    # echo "eg. easytrain ~ (train on $CUDA_VISIBLE_DEVICES)"
    # echo "eg. easytrain $CDX ~ & (后台运行)"
    # echo "$CDX:重定向, 不设置GPU_IDS则默认使用$CUDA_VISIBLE_DEVICES"
    exit

elif [ $1 == $CDX ];then
    cvd=$2
else
    cvd=$1
fi

_IFS=$IFS
IFS=','
gpus=($cvd)
IFS=$_IFS

if [ ${#gpus[@]} -eq 1 ];then
    if [ "$gpus" -gt -684123 ] 2>/dev/null ;then
        :
    else
        # not a number
        config=$cvd
        if [ $CUDA_VISIBLE_DEVICES ];then
            cvd=$CUDA_VISIBLE_DEVICES
        else
            cvd=0
        fi
        if [ $1 == $CDX ];then
            args=${@:3}
        else
            args=${@:2}
        fi
    fi
fi

if [ ! $config ];then
    if [ $1 == $CDX ];then
        config=$3
        args=${@:4}
    else
        config=$2
        args=${@:3}
    fi
fi

IFS=','
gpuids=($cvd)
IFS=$_IFS

gpus=${#gpuids[@]}

if [ $1 == $CDX ];then
    IFS='/'
    cfg=($config)
    IFS=$_IFS
    time=$(date "+%Y-%m-%d %H:%M:%S")
    #outf="$OutDir/$cvd-${cfg[-1]}[$time].txt"
    outf="$OutDir/last-$cvd.txt"
    bgargs="> '$outf' 2>&1 &"
fi

command="CUDA_VISIBLE_DEVICES=$cvd python $Train $config $args --gpus $gpus $bgargs"
echo $command
echo

if [ $1 == $CDX ];then
    if [ ! -d "$OutDir" ]; then
        echo "creat dir:$OutDir"
        mkdir $OutDir
    fi
    echo $command > "$outf" 2>&1

    CUDA_VISIBLE_DEVICES=$cvd python $Train $config $args --gpus $gpus \
        > "$outf" 2>&1
else
    CUDA_VISIBLE_DEVICES=$cvd python $Train $config $args --gpus $gpus 
fi


