#!/bin/bash

awk '{if(NF==2) print $1,$2,'$1'; else print $0}' | awk '{print $1,$2,($3>'$1')?'$1':$3}'
