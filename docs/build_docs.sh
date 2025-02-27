set -e  # Stop on first error

rm -rf build

# creates build/html/versions.py
python helpers/generate_versions.py

# get all branches and versions
versions=($(jq -r '.[].version' build/html/versions.json))
echo "Building docs for versions: ${versions[*]}"

for current_version in "${versions[@]}"; do

	echo "Version: $current_version"
	export current_version

	# double check that version exists
    if git rev-parse --verify "$current_version" >/dev/null 2>&1; then
		# load data from branch/version tag to folder in build/src
		(
			cd ..
			git archive --format=tar --prefix=docs/build/src/"$current_version"/ "$current_version" | tar -x
		)
		# build docs for this version
		(
			cd build/src/"$current_version"/docs
			make local
		)
		# move the generated html to the collective docs
		mv build/src/"$current_version"/docs/build/html/local build/html/"$current_version"
    else
        echo "Warning: Version $current_version not found. Skipping."
        continue
    fi

done
