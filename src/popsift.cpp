
#include "popsift.h"
#include <algorithm>
#include <mutex>

namespace pps{

PopSiftContext *ctx = nullptr;
std::mutex g_mutex;

PopSiftContext::PopSiftContext() : ps(nullptr){
    // Check if CUDA is working
    int currentDevice;
    if (cudaGetDevice( &currentDevice ) != 0){
        // Try resetting
        cudaDeviceReset();

        cudaError_t err;
        if ((err = cudaGetDevice( &currentDevice )) != 0){
            throw std::runtime_error("Cannot use CUDA device: " + std::string(cudaGetErrorString(err)));
        }
    }

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, false);
    config = new popsift::Config();
}

PopSiftContext::~PopSiftContext(){
    ps->uninit();
    delete ps;
    ps = nullptr;
    delete config;
    config = nullptr;
}

void PopSiftContext::setup(float peak_threshold, float edge_threshold, bool use_root, float downsampling){
    bool changed = false;
    if (this->peak_threshold != peak_threshold) { this->peak_threshold = peak_threshold; changed = true; }
    if (this->edge_threshold != edge_threshold) { this->edge_threshold = edge_threshold; changed = true; }
    if (this->use_root != use_root) { this->use_root = use_root; changed = true; }
    if (this->downsampling != downsampling) { this->downsampling = downsampling; changed = true; }

    if (changed){
        config->setThreshold(peak_threshold);
        config->setEdgeLimit(edge_threshold);
        config->setNormMode(use_root ? popsift::Config::RootSift : popsift::Config::Classic );
        config->setFilterSorting(popsift::Config::LargestScaleFirst);
        config->setMode(popsift::Config::OpenCV);
        config->setDownsampling(downsampling);
        // config->setOctaves(4);
        // config->setLevels(3);

        if (ps){
            delete ps;
            ps = nullptr;
        }
        ps = new PopSift(*config,
                    popsift::Config::ProcessingMode::ExtractingMode,
                    PopSift::ByteImages );
    }
}

PopSift *PopSiftContext::get(){
    return ps;
}

py::object popsift(pyarray_uint8 image,
                 float peak_threshold,
                 float edge_threshold,
                 bool use_root,
                 float downsampling) {
    if (!image.size()) return py::none();

    py::gil_scoped_release release;

    int width = image.shape(1);
    int height = image.shape(0);

    g_mutex.lock();
    if (!ctx) ctx = new PopSiftContext();
    ctx->setup(peak_threshold, edge_threshold, use_root, downsampling);
    std::unique_ptr<SiftJob> job(ctx->get()->enqueue( width, height, image.data() ));
    std::unique_ptr<popsift::Features> result(job->get());
    g_mutex.unlock();

    int numFeatures = result->getFeatureCount();

    popsift::Feature* feature_list = result->getFeatures();
    std::vector<float> points(4 * numFeatures);
    std::vector<float> desc(128 * numFeatures);

    for (size_t i = 0; i < numFeatures; i++){
        popsift::Feature pFeat = feature_list[i];

        for(int oriIdx = 0; oriIdx < pFeat.num_ori; oriIdx++){
            const popsift::Descriptor* pDesc = pFeat.desc[oriIdx];

            for (int k = 0; k < 128; k++){
                desc[128 * i + k] = pDesc->features[k];
            }

            points[4 * i + 0] = pFeat.xpos;
            points[4 * i + 1] = pFeat.ypos;
            points[4 * i + 2] = pFeat.sigma;
            points[4 * i + 3] = pFeat.orientation[oriIdx];
        }
    }

    py::gil_scoped_acquire acquire;
    py::list retn;
    retn.append(py_array_from_data(&points[0], numFeatures, 4));
    retn.append(py_array_from_data(&desc[0], numFeatures, 128));
    return retn;
}

bool fitsTexture(int width, int height, float downsampling){
    if (!ctx) ctx = new PopSiftContext();
    ctx->setup(0.06, 10, true, downsampling);

    PopSift::AllocTest a = ctx->get()->testTextureFit( width, height );
    return a == PopSift::AllocTest::Ok;
}

bool cudaIsAvailable(){
    int currentDevice;
    if (cudaGetDevice( &currentDevice ) != 0){
        // Try resetting
        cudaDeviceReset();

        if (cudaGetDevice( &currentDevice ) != 0){
            return false;
        }
    }

    return true;
}

std::tuple<size_t, size_t> getCudaMemoryInfo(){
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return std::make_tuple(free, total);
}

} // namespace pps
