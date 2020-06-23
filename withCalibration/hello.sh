#!/bin/bash
input="blank-codes.txt"
while IFS= read -r line
do
  mkdir blankAmazon/$line
  aws s3 cp s3://stardustathome.testbucket/real/$line ./calAmazon/$line --recursive
done < "$input"
