current_branch=$(git rev-parse --abbrev-ref HEAD)

versions=$(git tag --sort=-v:refname)
versions+=" main"

git checkout main -- helpers/ || echo "Using existing helpers"

python helpers/generate_versions.py

for current_version in $versions; do

	echo "Version: $current_version"
	export current_version

	git checkout "$current_version"

	rm -rf source/api/generated/*

	python helpers/create_nblinks.py
	python helpers/parse_classes.py

	sphinx-build --color -b html source -t "$current_version" build/html/${current_version} -v

done

git checkout "$current_branch"