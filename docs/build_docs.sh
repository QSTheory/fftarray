current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $current_branch"

versions=($(git tag --sort=-v:refname))
versions+=("main")

# git checkout main -- helpers/ || echo "Using existing helpers"

python helpers/generate_versions.py

for current_version in "${versions[@]}"; do

	echo "Version: $current_version"
	export current_version

	# Checkout the version safely (handle detached HEAD state for tags)
    if git rev-parse --verify "$current_version" >/dev/null 2>&1; then
        git checkout -b temp-$current_version "$current_version" || git checkout "$current_version"
    else
        echo "Warning: Version $current_version not found. Skipping."
        continue
    fi

	rm -rf source/api/generated/*

	python helpers/create_nblinks.py
	python helpers/parse_classes.py

	sphinx-build --color -b html source -t "$current_version" build/html/${current_version} -v

done

git checkout "$current_branch"
git branch -D $(git branch | grep temp-) || echo "No temporary branches to delete"
