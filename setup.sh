#!/bin/bash

rm -rf ./pgc_venv/
virtualenv --system-site-packages -p python3.8 ./pgc_venv
source ./pgc_venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
