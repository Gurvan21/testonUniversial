#!/bin/bash
# Larger EOT experiment, on ImageNet (protocol of the paper: X = 10 images per class, evaluation on other images):
# can a universal perturbation that knows the Gaussian noise defense (sigma = 20/255) break it?
# Gradient-based perturbations (--algo sgd), from the weakest to the strongest attack:
#   1. no EOT, xi = 10        control: does the gradient algorithm work without defense?
#   2. EOT, xi = 10           same budget as the paper
#   3. EOT, xi = 10, targeted towards "spider web" (the dominant class of resnet50)
#   4. EOT, xi = 16           larger budgets: which one breaks the defense?
#   5. EOT, xi = 24
# then each one is evaluated under the defense (eval_noise.py).
#
# Usage:   ./run_eot_imagenet.sh
#          MODEL=vit_b_16 EPOCHS=10 ./run_eot_imagenet.sh
#
# Can be relaunched at any time: the perturbations already computed are skipped, and an interrupted
# computation resumes from its checkpoint (saved after each epoch).

set -o pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

D=${DATA_DIR:-/Data/gurvan.nkouathio-kemnhou}
M=${MODEL:-resnet50}
SIGMA=${SIGMA:-20}
EPOCHS=${EPOCHS:-20}
O=results/eot_imagenet_$M
mkdir -p "$O"

common="-m $M -t $D/imagenet_val/X --cache $D/cache/imagenet_val_X.pt --algo sgd --epochs $EPOCHS
        --sgd_batch_size 32 --eot_samples 4"

run() {
    name=$1; shift
    if [ -f "$O/$name.pt" ]; then
        echo "=== $name : déjà calculée, on passe ==="
        return
    fi
    echo "=== $name : calcul de la perturbation ($(date)) ==="
    python demo.py $common -o "$O/$name.pt" "$@" 2>&1 | tee -a "$O/log_$name.txt" || exit 1
}

run sgd_xi10 --xi 10
run sgd_eot${SIGMA}_xi10 --xi 10 --noise_std "$SIGMA"
run sgd_eot${SIGMA}_xi10_spiderweb --xi 10 --noise_std "$SIGMA" --target "spider web"
run sgd_eot${SIGMA}_xi16 --xi 16 --noise_std "$SIGMA"
run sgd_eot${SIGMA}_xi24 --xi 24 --noise_std "$SIGMA"

for f in "$O"/sgd_*.pt; do
    name=$(basename "$f" .pt)
    case $name in *_checkpoint) continue ;; esac
    [ -f "$O/noise_$name.txt" ] && continue
    echo "=== $name : évaluation sous la défense ($(date)) ==="
    python eval_noise.py -m "$M" -p "$f" -v "$D/imagenet_val/eval" --max_per_class 5 --sigmas 0,10,20,40 \
        --out "$O/noise_$name.txt" 2>&1 | tail -n 20 || exit 1
done

echo "=== Terminé ($(date)) ==="
