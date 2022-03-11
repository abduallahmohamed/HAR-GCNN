wget -O data.zip 'https://utexas.box.com/shared/static/nx1n93pwhtad61834yr7u7d6330qqwxt.zip'
unzip -q data.zip
rm -rf data.zip
mv dataset/* .
rm -rf dataset
