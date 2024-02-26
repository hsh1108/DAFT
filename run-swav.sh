run_name="1 2 3"
for r in $run_name
do
    # FT
    python ft_sgd.py --config configs_swav/cifar-ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/entity-ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/domainnet-ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/fmow-ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/living-ft.yaml  --run_num $r

    # LP
    python extract_features.py --config configs_swav/cifar-lp.yaml --run_num $r
    python lp_log_reg.py --config configs_swav/cifar-lp.yaml  --run_num $r
    python extract_features.py --config configs_swav/entity-lp.yaml  --run_num $r
    python lp_log_reg.py --config configs_swav/entity-lp.yaml  --run_num $r
    python extract_features.py --config configs_swav/domainnet-lp.yaml  --run_num $r
    python lp_log_reg.py --config configs_swav/domainnet-lp.yaml  --run_num $r
    python extract_features.py --config configs_swav/living-lp.yaml  --run_num $r
    python lp_log_reg.py --config configs_swav/living-lp.yaml  --run_num $r
    python extract_features.py --config configs_swav/fmow-lp.yaml  --run_num $r
    python lp_log_reg.py --config configs_swav/fmow-lp.yaml  --run_num $r
    
    # LP-FT (Before running this, make sure to run the above LP scripts)
    python ft_sgd.py --config configs_swav/cifar-lp_ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/entity-lp_ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/domainnet-lp_ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/living-lp_ft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/fmow-lp_ft.yaml  --run_num $r

    # DAFT
    python ft_sgd.py --config configs_swav/cifar-daft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/entity-daft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/domainnet-daft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/fmow-daft.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/living-daft.yaml  --run_num $r
    # daft_2 is for finding the optimized lr for head layer.
    # After find optimized lr for feature extractor with daft, run daft_2.
    # You should specify the lr of feature extractor in the daft_2 file.
    python ft_sgd.py --config configs_swav/cifar-daft_2.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/entity-daft_2.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/domainnet-daft_2.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/fmow-daft_2.yaml  --run_num $r
    python ft_sgd.py --config configs_swav/living-daft_2.yaml  --run_num $r

done
