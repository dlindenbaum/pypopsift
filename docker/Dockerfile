# manylinux2014_x86_64_cuda_12_3
FROM quay.io/pypa/manylinux2014_x86_64

RUN yum install -y yum-utils
RUN yum install -y wget
RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-rhel7-12-3-local-12.3.0_545.23.06-1.x86_64.rpm
RUN rpm -i cuda-repo-rhel7-12-3-local-12.3.0_545.23.06-1.x86_64.rpm
RUN yum clean all
RUN yum -y install cuda-toolkit-12-3

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV CUDADIR=/usr/local/cuda
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]