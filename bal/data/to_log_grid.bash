#!/bin/bash 

awk 'BEGIN{last=-42;}{if(NR==1 || $1==0 || $2==0) next; if(last!=-42 && last!=$1) {printf "\n";} last=$1; print(log($1)/log(10), log($2)/log(10), $3);}'
