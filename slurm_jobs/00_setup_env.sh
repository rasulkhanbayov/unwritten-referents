#!/bin/bash
# Run this ONCE, interactively, on one of the SHARED PHYSICAL SERVERS
# (10.4.64.27 or 10.4.64.28), NOT on panther-login -- the login node kills
# stray long-running user processes (python, conda installers included), per
# the Panther End User Guide. SSH there first:
#   ssh <username>@10.4.64.27
# then run this script from the same path (home directory is shared/symlinked
# across panther-login, the shared servers, and compute nodes, so no copying
# is needed). Modeled on the equivalent setup script in
# .../Rasul_research_ideas (2)/3-ACL/generator_build/slurm_jobs/00_setup_env.sh
# from this user's prior project on the same account -- same conda pattern,
# same "why cu124 not cu121" reasoning applies here too (see below).
#
# Safe to re-run: the conda install step is skipped if already present, and
# the env-create step is skipped if the env already exists.
set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="skc_iclr27"
# Repo root: $SKC_REPO if set, else walk up from where sbatch was invoked (or this
# script's dir when run with plain bash) until src/skc_data.py is found. Do NOT use
# dirname "$0" alone: under sbatch the script runs from a spool copy, not the repo.
_d="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
while [ "$_d" != "/" ] && [ ! -f "$_d/src/skc_data.py" ]; do _d="$(dirname "$_d")"; done
REPO="${SKC_REPO:-$_d}"
[ -f "$REPO/src/skc_data.py" ] || { echo "Cannot locate repo root; export SKC_REPO=/path/to/repo" >&2; exit 1; }

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "== conda not found at $CONDA_DIR -- installing Miniconda =="
    TMP_INSTALLER="$HOME/miniconda_installer.sh"
    curl -fsSL -o "$TMP_INSTALLER" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash "$TMP_INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$TMP_INSTALLER"
    echo "== Miniconda installed to $CONDA_DIR =="
else
    echo "== conda already installed at $CONDA_DIR =="
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

# Anaconda's default channels now require Terms of Service acceptance before
# any env can be created or solved against them -- a policy change on
# Anaconda's side. Accept once, non-interactively, so env creation below does
# not fail with CondaToSNonInteractiveError.
echo "== accepting Anaconda channel Terms of Service (one-time) =="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "== creating conda env $ENV_NAME (python 3.11) if missing =="
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

echo "== installing packages from requirements.txt (torch first, pinned) =="
pip install --upgrade pip
# skc_runbook.txt section 0 pins torch>=2.4, transformers>=4.44. Use the
# cu124 wheel index rather than cu121: on the prior project on this same
# account, a plain 'pip install torchvision' with no pin silently upgraded
# torch itself (2.6.0 -> 2.13.0) as a side effect of resolving torchvision's
# OWN dependency on the latest torch. Pin torch AND torchvision together in
# ONE pip command to avoid repeating that.
#
# CORRECTION, found on this project's first real GPU job: the cu124 wheel
# bundles its own complete CUDA 12.4 runtime and needs NOTHING from a system
# `module load cuda...` toolkit -- but that module load is not merely
# unnecessary, it is ACTIVELY HARMFUL if any sbatch script loads one anyway.
# `module load cuda12.1/toolkit/12.1.1` was tried in every slurm_jobs/*.sh
# script and produced `ImportError: ...libcusparse.so.12: undefined symbol:
# __nvJitLinkComplete_12_4` the moment `import torch` ran on a real A100
# node: the module prepends an OLDER cusparse/nvJitLink onto
# LD_LIBRARY_PATH, ahead of torch's own bundled cu124 libs, and the
# dynamic linker resolves the older, incompatible one first. Every
# slurm_jobs/*.sh script has since had its `module load cuda...` line
# removed for exactly this reason. Do not add one back "to be safe" --
# nothing in this pipeline compiles a custom CUDA extension that would need it.
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu124
pip install -r "$REPO/requirements.txt"

# circuit-tracer (C4 attribution graphs) installs from source per
# requirements.txt's comment; NOT pip-installable from PyPI as of this
# writing. Only needed for skc_run_e2_c4_graphs.py (not yet written -- see
# the execution plan) -- skip it here and install separately when that
# script exists, so a circuit-tracer install failure does not block every
# other experiment from getting a working environment.
echo
echo "== NOT installing circuit-tracer here (needed only for C4 / attribution graphs). =="
echo "   When ready for C4, follow https://github.com/safety-research/circuit-tracer "
echo "   inside this same '$ENV_NAME' env."

echo
echo "== done. =="
echo "IMPORTANT: run 'conda init bash' once and re-login, OR always source"
echo "  $CONDA_DIR/etc/profile.d/conda.sh"
echo "before 'conda activate $ENV_NAME' in any NEW shell (including inside"
echo "sbatch scripts) -- conda activate does not work in a fresh non-interactive"
echo "shell without sourcing conda.sh first. The sbatch scripts in this"
echo "directory already do this for you."
echo
echo "Verify GPU visibility works from THIS shell (a shared server, may or may"
echo "not have a free GPU right now -- the real check that matters happens"
echo "inside the actual sbatch job on a Panther A100/H200 compute node):"
echo "  conda activate $ENV_NAME && python3 -c 'import torch; print(torch.cuda.is_available())'"
echo
echo "Also verify the two OFFLINE self-tests still pass in this env before submitting"
echo "anything to Slurm (they need only numpy, no GPU):"
echo "  cd $REPO/src && python3 skc_metrics.py --smoke-test && python3 skc_pipeline_test.py"
