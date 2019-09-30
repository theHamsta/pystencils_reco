#! /bin/sh
#
# update_notebooks.sh
# Copyright (C) 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
#


jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --inplace --execute docs/notebooks/*.ipynb

