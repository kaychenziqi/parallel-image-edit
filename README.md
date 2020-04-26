# parallel-image-edit

## PatchMatch

### Compile

```
make all
make seq
make omp
make cuda
```

### Run

```
./PatchMatchSeq -i img/Cloth2-view1.jpg -s img/Cloth1-view1.jpg -o output/Cloth2-view1.jpg -p 5
```

- `-i`: input file
- `-o`: output file
- `-s`: source file
- `-w`: resize width, default is input image width
- `-h`: resize height, default is input image height
- `-p`: half patch size