for current_version in 'public' 'development'; do
	export current_version
	./build_api.sh
    sphinx-build --color -b html source -t "$current_version" build/html/${current_version}
	# sphinx-build -b html source -t "$current_version" build/html/${current_version}
done
