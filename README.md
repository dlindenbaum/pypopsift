# pypopsift
CUDA accelerated SIFT in Python

This library is a wrapper around [PopSift](https://github.com/alicevision/popsift) to compute SIFT keypoints and descriptors on the GPU using CUDA. It's written to be a drop-in replacement for existing OpenCV functions such as `cv2.FeatureDetector_create('SIFT')` and ` cv2.DescriptorExtractor_create('SIFT')`.

## Building

Requirements:
 * CUDA Toolkit >= 10
    * https://developer.nvidia.com/cuda-downloads
 * CMake >= 3.14
 * A C++11 capable compiler (g++-7, gcc-7)
 * python3.10-dev

```
git clone --recurse-submodules git@github.com:braincorp/pypopsift.git
cd pypopsift && mkdir build && cd build
cmake ..
make -j8
```

To install the Python package:

```
cd pypopsift
pip install .
```

## Usage

```
python example.py
```

## License

Mozilla Public License 2.0

## Acknowledgements

```
@inproceedings{Griwodz2018Popsift,
	 author = {Griwodz, Carsten and Calvet, Lilian and Halvorsen, P{\aa}l},
	 title = {Popsift: A Faithful SIFT Implementation for Real-time Applications},
	 booktitle = {Proceedings of the 9th {ACM} Multimedia Systems Conference},
	 series = {MMSys '18},
	 year = {2018},
	 isbn = {978-1-4503-5192-8},
	 location = {Amsterdam, Netherlands},
	 pages = {415--420},
	 numpages = {6},
	 doi = {10.1145/3204949.3208136},
	 acmid = {3208136},
	 publisher = {ACM},
	 address = {New York, NY, USA},
}
```
