# mnistasm
mnistasm is an x64 assembly implementation of an [mnist](https://en.wikipedia.org/wiki/MNIST_database) digit classifier. it predicts a given digit's label by picking the label of the nearest neighbor in by cosine similarity.

the original mnist dataset contains 28x28 grayscale images, but they're really 20x20 images centered by mass and padded by 4 on each side. the preprocess step extracts this 20x20 center, flattens it, and rounds pixel values to 0 or 1.

vectors thus occupy 400 bits, and with the power of avx-512 we can store a vector in a single zmm register. this makes all of our 400-dimensional vector operations take only a handful of clock cycles. all necessary vector math essentially boils down to bitwise ands, popcounts, and reciprocal sqrts.

## preprocess
i've included `train.dat` and `test.dat` for ease of use, but you can generate these files from your own training and test sets with `./preprocess.py --train <training-set-path> --test <test-set-path>`, assuming they're in the same format as mnist's.

## build and run
`nasm -f elf64 main.asm && ld main.o && ./a.out`

the classifier will run on each vector in `./test.dat`, then report its accuracy.
