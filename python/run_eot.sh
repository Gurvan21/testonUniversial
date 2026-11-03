#!/bin/bash
# First EOT experiment: resnet50, 500 Imagenette images, Gaussian noise defense sigma = 20/255.
# Three perturbations computed on the same images (without EOT, DeepFool + EOT, gradient + EOT),
# then each one evaluated under the noise defense.
#
# Usage:   ./run_eot.sh
#          SIGMA=10 SGD_BATCH=8 ./run_eot.sh      (SGD_BATCH x EOT_SAMPLES <= 16 on a 2 GB GPU)
#
# Can be relaunched at any time: the perturbations already computed are skipped, and an interrupted
# computation resumes from its checkpoint.

set -o pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

M=${MODEL:-resnet50}
SIGMA=${SIGMA:-20}
EOT_SAMPLES=${EOT_SAMPLES:-4}
SGD_BATCH=${SGD_BATCH:-4}
T=data/imagenette2-320/train
V=data/imagenette2-320/val
C=data/cache_imagenette_500.pt
O=results/eot
mkdir -p "$O"

common="-m $M -t $T --num_imgs_per_class 50 --cache $C"

run() {
    name=$1; shift
    if [ -f "$O/$name.pt" ]; then
        echo "=== $name : déjà calculée, on passe ==="
        return
    fi
    echo "=== $name : calcul de la perturbation ($(date)) ==="
    python demo.py $common -o "$O/$name.pt" "$@" 2>&1 | tee -a "$O/log_$name.txt" || exit 1
}

run ${M}_df --max_passes 10
run ${M}_df_eot$SIGMA --noise_std "$SIGMA" --eot_samples "$EOT_SAMPLES" --max_passes 5
run ${M}_sgd_eot$SIGMA --algo sgd --noise_std "$SIGMA" --eot_samples "$EOT_SAMPLES" --sgd_batch_size "$SGD_BATCH" --epochs 10

for name in ${M}_df ${M}_df_eot$SIGMA ${M}_sgd_eot$SIGMA; do
    echo "=== $name : évaluation sous la défense ($(date)) ==="
    python eval_noise.py -m "$M" -p "$O/$name.pt" -v "$V" --sigmas 0,10,20,40 --max_per_class 100 \
        --out "$O/noise_$name.txt" 2>&1 | tail -n 20 || exit 1
done

echo "=== Terminé ($(date)) ==="
