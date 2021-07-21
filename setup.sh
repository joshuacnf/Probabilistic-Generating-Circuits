#!/bin/bash

if [ "$1" = "" ]; then
    env_path="."
else
    env_path=$1
fi

rm -rf $env_path/pgc_venv/
virtualenv --system-site-packages -p python3.8 $env_path/pgc_venv
source $env_path/pgc_venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
