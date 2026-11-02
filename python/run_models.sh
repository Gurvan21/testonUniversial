#!/bin/bash
# Compute and evaluate the universal perturbation of several models on ImageNet (protocol of the paper:
# X = 10 images per class, evaluation on the 40 000 other validation images).
#
# Usage:   ./run_models.sh convnext_base swin_b vit_b_16 maxvit_t
#          DATA_DIR=/Data/... MAX_ITER_DF=20 ./run_models.sh vit_b_16
#
# Can be relaunched at any time: the models already evaluated are skipped, and the computation of an
# interrupted model resumes from its checkpoint (data/checkpoint_<model>.pt).

set -o pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
# Print the output of python immediately (otherwise it is buffered when piped to tee)
export PYTHONUNBUFFERED=1

D=${DATA_DIR:-/Data/gurvan.nkouathio-kemnhou}
MAX_ITER_DF=${MAX_ITER_DF:-10}

for m in "$@"; do
    out=results/imagenet_${m}_10k
    if [ -f "$out/eval_$m.txt" ]; then
        echo "=== $m : déjà évalué ($out), on passe ==="
        continue
    fi
    mkdir -p "$out"

    echo "=== $m : calcul de la perturbation ($(date)) ==="
    python demo.py -m "$m" -t "$D/imagenet_val/X" --cache "$D/cache/imagenet_val_X.pt" \
        --max_iter_df "$MAX_ITER_DF" 2>&1 | tee -a "$out/log_demo.txt" || exit 1

    echo "=== $m : évaluation ($(date)) ==="
    python evaluate.py -m "$m" -v "$D/imagenet_val/eval" --out "$out" 2>&1 | tee "$out/log_eval.txt" || exit 1
    cp "data/universal_$m.pt" "$out/"
done

echo "=== Terminé ($(date)) ==="
