#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <math.h>

#define CHECKCUDA(a) do {\
    if (cudaSuccess != (a)) {\
        printf("Failed!\n"); \
        abort(); }\
    } while (0);

#define CHECKCUDNN(a) do {\
    if (CUDNN_STATUS_SUCCESS != (a)) {\
        printf("Failed!\n"); \
        abort(); }\
    } while (0);

/*
void generate_rand_num(float* ptr, int length) {
    for (int i = 0; i < length; ++i){
        float value = ((float) rand()) / ((float) RAND_MAX);
        ptr[i] = value;
    }
}*/

int main(int argc, char** argv){
    /*
     * Usage: ./main <IN> <IC> <IH> <IW> <OC> <FH> <FW> <OH> <OW>
     *              <pad_h> <pad_w> <strd_h> <strd_w> <mode> <conv_format>
     */
    if (argc != 16){
        printf("Failed!\n");
        return 0;
    }

    int IN = atoi(argv[1]);
    int ON = IN;
    int IC = atoi(argv[2]);
    int IH = atoi(argv[3]);
    int IW = atoi(argv[4]);
    int OC = atoi(argv[5]);
    int FH = atoi(argv[6]);
    int FW = atoi(argv[7]);
    int OH = atoi(argv[8]);
    int OW = atoi(argv[9]);
    int pad_h = atoi(argv[10]);
    int pad_w = atoi(argv[11]);
    int strd_h = atoi(argv[12]);
    int strd_w = atoi(argv[13]);
    int mode = atoi(argv[14]);
    int conv_format = atoi(argv[15]);

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECKCUDNN(cudnnCreate(&handle));
    CHECKCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECKCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECKCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECKCUDNN(cudnnCreateFilterDescriptor(&wDesc));

    CHECKCUDNN(cudnnSetTensor4dDescriptor(xDesc, (cudnnTensorFormat_t)conv_format,
            CUDNN_DATA_FLOAT, IN, IC, IH, IW));
    CHECKCUDNN(cudnnSetTensor4dDescriptor(yDesc, (cudnnTensorFormat_t)conv_format,
            CUDNN_DATA_FLOAT, ON, OC, OH, OW));
    CHECKCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w,
                strd_h, strd_w, 1, 1, (cudnnConvolutionMode_t)mode, CUDNN_DATA_FLOAT));
    CHECKCUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
                (cudnnTensorFormat_t)conv_format, OC, IC, FH, FW));

    cudnnConvolutionFwdAlgo_t curr_algo;
    CHECKCUDNN(cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc,
                yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &curr_algo));
    printf("Preferred Algo: %d\n", (int)curr_algo);

    int maxAlgoCount;
    int returnedAlgoCount;

    CHECKCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgoCount));
    cudnnConvolutionFwdAlgoPerf_t* perfResults = (cudnnConvolutionFwdAlgoPerf_t*)malloc(
            sizeof(cudnnConvolutionFwdAlgoPerf_t) * maxAlgoCount);

    CHECKCUDNN(cudnnFindConvolutionForwardAlgorithm(handle,
                xDesc, wDesc, convDesc, yDesc, maxAlgoCount, &returnedAlgoCount, perfResults))

    for (int i = 0; i < returnedAlgoCount; ++i)
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
            printf("Algo: %d, Time: %.6f ms, Workspace: %zu bytes\n",
                    (int)perfResults[i].algo, perfResults[i].time, perfResults[i].memory);
        }

    CHECKCUDNN(cudnnDestroy(handle));
    CHECKCUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECKCUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECKCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECKCUDNN(cudnnDestroyFilterDescriptor(wDesc));

    free(perfResults);

    return 0;
}
