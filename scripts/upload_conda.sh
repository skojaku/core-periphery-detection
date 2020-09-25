# run conda skeleton pypi cpnet
pkg='cpnet'
array=( 3.6 3.7 3.8 )

tmpdir=Bbsca4Ne
metafile=scripts/meta.yaml

conda_folder=$HOME/opt/anaconda3/conda-bld
arch=osx-64

rm -r $tmpdir
mkdir $tmpdir 
mkdir $tmpdir"/"$pkg
cp $metafile $tmpdir"/"$pkg
cd $tmpdir

ls

echo "Building conda package ..."
# building conda packages
for i in "${array[@]}"
do
	conda-build --python $i $pkg --numpy 1.16.0 -c conda-forge
done

platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
find $conda_folder/$arch/ -name *.tar.bz2 | while read file
do
    echo $file
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file  -o $conda_folder
    done
    
done

# upload packages to conda
find $conda_folder/ -name *.tar.bz2 | while read file
do
    echo $file
    anaconda upload $file --force
done

cd .. && rm -rf $tmpdir
