#include <stdio.h>
#include <assert.h>

#define TRAINING_IMAGES_NUM 50000

int reverse(int num) {
    return ((num & 0xff) << 24) |
           ((num & 0xff00) << 8) |
           ((num & 0xff0000) >> 8) |
           ((num & 0xff000000) >> 24);
}

void transform(const char* images_fname, const char* labels_fname, const char* out_fname) {
    // ---------------------------------------------------------------
    //  images file's format
    // ===============================================================
    //
    //  [offset] [type]          [value]          [description] 
    //  0000     32 bit integer  0x00000803(2051) magic number 
    //  0004     32 bit integer  60000            number of images 
    //  0008     32 bit integer  28               number of rows 
    //  0012     32 bit integer  28               number of columns 
    //  0016     unsigned byte   ??               pixel 
    //  0017     unsigned byte   ??               pixel 
    //  ........ 
    //  xxxx     unsigned byte   ??               pixel
    //
    // ===============================================================
    //  Pixels are organized row-wise. Pixel values are 0 to 255.
    //  0 means background (white), 255 means foreground (black).
    // ---------------------------------------------------------------

    FILE* fimages = fopen(images_fname, "rb");
    int images_magic, images_num, row_num, col_num;

    fread(&images_magic, 1, 4, fimages);
    fread(&images_num, 1, 4, fimages);
    fread(&row_num, 1, 4, fimages);
    fread(&col_num, 1, 4, fimages);

    images_magic = reverse(images_magic);
    assert(images_magic == 0x0803);
    images_num = reverse(images_num);
    row_num = reverse(row_num);
    col_num = reverse(col_num);

    // ---------------------------------------------------------------
    //  labels file's format
    // ===============================================================
    //
    // [offset] [type]          [value]          [description] 
    // 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
    // 0004     32 bit integer  60000            number of items 
    // 0008     unsigned byte   ??               label 
    // 0009     unsigned byte   ??               label 
    // ........ 
    // xxxx     unsigned byte   ??               label
    //
    // ===============================================================
    // The labels values are 0 to 9.
    // ---------------------------------------------------------------

    FILE* flabels = fopen(labels_fname, "rb");
    int labels_magic, labels_num;

    fread(&labels_magic, 1, 4, flabels);
    fread(&labels_num, 1, 4, flabels);

    labels_magic= reverse(labels_magic);
    assert(labels_magic == 0x0801);

    labels_num = reverse(labels_num);
    assert(labels_num == images_num);

    FILE* fout = fopen(out_fname, "w");
    unsigned char pixel, label;
    for (int i = 0; i < images_num; ++i) {
        fread(&label, 1, 1, flabels);
        fprintf(fout, "%u", label);

        for (int r = 0; r < row_num; ++r) {
            for (int c = 0; c < col_num; ++c) {
                fread(&pixel, 1, 1, fimages);
                fprintf(fout, ",%u", pixel);
            }
        }
        fprintf(fout, "\n");
    }

    fclose(fimages);
    fclose(flabels);
    fclose(fout);
}

void split_train_data(const char* all_fname,
                      const char* training_fname,
                      const char* validation_fname) {
    FILE* ftraining = fopen(training_fname, "w");
    FILE* fvalidation = fopen(validation_fname, "w");
    FILE* fall = fopen(all_fname, "r");

    char line[4 * 28 * 28 + 1 + 1];
    for (int i = 0; i < TRAINING_IMAGES_NUM; ++i) {
        fgets(line, sizeof(line), fall);
        fprintf(ftraining, "%s", line);
    }

    while (fgets(line, sizeof(line), fall) != NULL) {
        fprintf(fvalidation, "%s", line);
    }

    fclose(ftraining);
    fclose(fvalidation);
    fclose(fall);
}

int main() {
    transform("./train-images-idx3-ubyte",
              "./train-labels-idx1-ubyte",
              "../training.all");

    split_train_data("../training.all", "../training", "../validation");

    transform("./t10k-images-idx3-ubyte",
              "./t10k-labels-idx1-ubyte",
              "../testing");

    return 0;
}
