#!/bin/sh

# setup git pre-commit hook to make an automatic verification before each
# commit
echo ../.git/hooks/
cd ../.git/hooks/
ln -sv ../../scripts/pre-commit .
