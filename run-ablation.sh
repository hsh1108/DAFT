
run_name="1 2 3"
for r in $run_name
do    
    python ft_sgd.py --config configs_moco/cifar-lp_ft.yaml  --run_num $r --use_bn_conversion
    python ft_sgd.py --config configs_moco/cifar-daft.yaml  --run_num $r --ablation 'only_bn_conversion'
    python ft_sgd.py --config configs_moco/cifar-daft.yaml  --run_num $r --ablation 'only_fast_headtrain'
    python ft_sgd.py --config configs_moco/cifar-lp_ft.yaml  --run_num $r --use_bn_conversion

    python ft_sgd.py --config configs_moco/cifar-lp_ft.yaml  --run_num $r --use_bn_conversion
    python ft_sgd.py --config configs_moco/cifar-daft.yaml  --run_num $r --ablation 'only_bn_conversion'
    python ft_sgd.py --config configs_moco/cifar-daft.yaml  --run_num $r --ablation 'only_fast_headtrain'
    python ft_sgd.py --config configs_moco/cifar-lp_ft.yaml  --run_num $r --use_bn_conversion

    python ft_sgd.py --config configs_moco/cifar-lp_ft.yaml  --run_num $r --use_bn_conversion
    python ft_sgd.py --config configs_moco/cifar-daft.yaml  --run_num $r --ablation 'only_bn_conversion'
    python ft_sgd.py --config configs_moco/cifar-daft.yaml  --run_num $r --ablation 'only_fast_headtrain'
    python ft_sgd.py --config configs_moco/cifar-lp_ft.yaml  --run_num $r --use_bn_conversion

done
