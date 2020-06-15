#!/bin/bash
input="am-codes.txt"
while IFS= read -r line
do
  mkdir calAmazon/$line
  aws s3 cp s3://stardustathome.testbucket/real/$line ./calAmazon/$line --recursive
done < "$input"
