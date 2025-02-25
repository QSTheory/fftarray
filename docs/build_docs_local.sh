set -e  # Stop on first error
python helpers/generate_versions.py
python helpers/create_nblinks.py
python helpers/parse_classes.py
rm -rf source/api/generated/*
sphinx-build --color -b html source build/html/local -v
