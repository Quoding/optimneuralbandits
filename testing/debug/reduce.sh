for d in $(find runs -maxdepth 2 -mindepth 2 -type d)
do
  #Do something, the directory is accessible with $d:
  tb-reducer ${d}/*  -o ${d}_reduced -r mean,std,min,max --lax-steps
done



