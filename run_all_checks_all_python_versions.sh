# This script aims to run the whole test suite locally to prevent surprises in CI.
pixi run -e doc doc_local && continue || exit 1
pixi run -e doc doc_all_versions && continue || exit 1
for pyenv in check310 check311 check312 check313; do
  (command pixi run -e $pyenv check_all) && continue || exit 1
done