#!/bin/bash

echo -e "\033[0;32mDeploying documentation to GitHub...\033[0m"

#!/bin/bash
git checkout master
cd docs
make html
cd ..
git checkout gh-pages
cp -r docs/build/html/* .
git add .
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "Update documentation"
git push -u comphy gh-pages
git checkout master