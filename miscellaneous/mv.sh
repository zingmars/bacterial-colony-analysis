a=$1
files=`ls . -I mv.sh`
for i in $files; do
  new=$(printf "%04d.jpg" "$a") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let a=a+1
done
