#! /bin/sh 
ls | sed 's/\t/\n/g' | grep -o '^auto4_[0-9]\+_[0-9]\+_[0-9]\+_' | sort | uniq > list.in

while IFS= read -r line
do
	sh draw.sh "$line"
done <"list.in"
