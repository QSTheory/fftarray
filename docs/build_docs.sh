for current_version in 'public' 'development'; do
	export current_version
	python create_nblinks.py
    sphinx-build --color -b html source -t "$current_version" build/html/${current_version} -v
done
