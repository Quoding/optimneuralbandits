for d in $(find runs -maxdepth 2 -mindepth 2 -type d)
do
  #Do something, the directory is accessible with $d:
  echo $d
done
