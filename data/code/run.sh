curl -# -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -# -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -# -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -# -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz

gcc transform.c -o transform -Wall && ./transform && rm -f transform

rm -f t10k-images-idx3-ubyte
rm -f t10k-labels-idx1-ubyte
rm -f train-images-idx3-ubyte
rm -f train-labels-idx1-ubyte
