import os
import json
import subprocess

BUILD_DIR = "build/html/"
os.system(f"mkdir -p {BUILD_DIR}")

branches = ["main"]

# Get all version directories inside build/html/
versions = sorted(
    subprocess.check_output(["git", "tag"], text=True).strip().split("\n"),
    reverse=True  # Show latest version first
)

exclude_versions = ["0.4a0"]

formatted_versions = []
for v in branches+versions:
    if v not in exclude_versions:
        version_entry = {
            "version": v,
            # "url": f"https://QSTheory.github.io/fftarray/{version}/"
            "url": f"http://localhost:8000/{v}/"
        }

        if v in branches:
            version_entry["name"] = f"dev ({v})"
        elif v == versions[0]:
            version_entry["name"] = f"{v} (stable)"

        formatted_versions.append(version_entry)

# Save the versions as a JSON file in the root of the build
with open(BUILD_DIR + "versions.json", "w") as f:
    json.dump(formatted_versions, f, indent=4)
