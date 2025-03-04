pixi run -e doc doc  && continue || exit 1
for pyenv in check310 check311 check312 check313; do
  (command pixi run -e $pyenv check_all) && continue || exit 1
done