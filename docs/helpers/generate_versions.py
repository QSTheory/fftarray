import json
import subprocess

# Get all version directories inside build/html/
versions = sorted(
    ["main"] + subprocess.check_output(["git", "tag"], text=True).strip().split("\n"),
    reverse=True  # Show latest version first
)

exclude_versions = ["0.4a0"]

formatted_versions = []
for version in versions:
    if version not in exclude_versions:
        version_entry = {
            "version": version,
            # "url": f"https://QSTheory.github.io/fftarray/{version}/"
            "url": f"http://localhost:8000/{version}/"
        }

        if version == versions[0]:
            version_entry["name"] = f"dev"

        if version == versions[1]:
            version_entry["name"] = f"{version} (stable)"

        formatted_versions.append(version_entry)


# Save the versions as a JSON file in the root of the build
with open("build/html/versions.json", "w") as f:
    json.dump(formatted_versions, f, indent=4)
