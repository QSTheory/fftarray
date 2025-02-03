for current_version in 'public' 'development'; do
	export current_version
	python helpers/create_nblinks.py
	python helpers/parse_classes.py
    sphinx-build --color -b html source -t "$current_version" build/html/${current_version} -v
done
