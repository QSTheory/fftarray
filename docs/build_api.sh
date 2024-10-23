rm -rf source/api

exclude=$(<exclude.txt)
exclude=$(echo "$exclude" | tr "\n" " ")
echo "excluding: $exclude"

private=$(<private.txt)
private=$(echo "$private" | tr "\n" " ")
echo "private: $private"

public="public"
if [ "$current_version" == "$public" ]
then
	sphinx-apidoc -o source/api ../fftarray $exclude $private -P -M
else
	sphinx-apidoc -o source/api ../fftarray $exclude -P -M
fi

python create_nblinks.py
